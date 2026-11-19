// Feature gather on the device: out[i, :] = feat[idx[i], :].
//
// The feature table can live in any USM kind, which is the question this measures on an
// integrated GPU (same physical RAM as the CPU):
//   device  malloc_device — the driver may place/compress it for the GPU
//   host    malloc_host   — CPU memory the GPU reads directly (zero copy)
//   shared  malloc_shared — migrates on demand
// The table is split into row chunks that each fit the device's largest single allocation
// (1.07 GB on the laptop iGPU), so feature tables of any size work.
// Mapping: one work-item per 16-byte chunk of a row, so a sub-group reads consecutive
// chunks of the same row (coalesced) and rows are independent.
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gnnsample {

enum class UsmKind { device, host, shared };

inline UsmKind parse_usm(const std::string& s) {
  if (s == "device") return UsmKind::device;
  if (s == "host") return UsmKind::host;
  if (s == "shared") return UsmKind::shared;
  throw std::invalid_argument("usm kind must be device, host or shared");
}

template <class T>
T* usm_alloc(sycl::queue& q, size_t n, UsmKind kind) {
  switch (kind) {
    case UsmKind::device: return sycl::malloc_device<T>(n, q);
    case UsmKind::host: return sycl::malloc_host<T>(n, q);
    default: return sycl::malloc_shared<T>(n, q);
  }
}

class FeatureTable {
 public:
  // Copies a row-major [rows, dim] float table into USM of the requested kind.
  // dim must be a multiple of 4 (16-byte chunks); pad on the caller side otherwise.
  FeatureTable(sycl::queue& q, const float* data, int64_t rows, int64_t dim, UsmKind kind)
      : q_(q), rows_(rows), dim_(dim) {
    if (dim % 4) throw std::invalid_argument("feature dim must be a multiple of 4");
    // The driver's reported limit grows with UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS, but
    // host USM allocations above ~1 GB still fail on the iGPU, so cap chunks at 512 MB.
    const uint64_t max_bytes = std::min<uint64_t>(
        q.get_device().get_info<sycl::info::device::max_mem_alloc_size>(), uint64_t{512} << 20);
    rows_per_chunk_ = std::max<int64_t>(1, max_bytes / (sizeof(float) * dim));
    for (int64_t r = 0; r < rows; r += rows_per_chunk_) {
      const int64_t n = std::min(rows_per_chunk_, rows - r);
      float* p = usm_alloc<float>(q, static_cast<size_t>(n) * dim, kind);
      if (!p) throw std::runtime_error("USM allocation failed");
      q.memcpy(p, data + r * dim, sizeof(float) * n * dim);
      chunks_.push_back(p);
    }
    chunk_ptrs_ = sycl::malloc_device<float*>(chunks_.size(), q);
    q.memcpy(chunk_ptrs_, chunks_.data(), chunks_.size() * sizeof(float*)).wait();
  }
  ~FeatureTable() {
    for (float* p : chunks_) sycl::free(p, q_);
    sycl::free(chunk_ptrs_, q_);
  }
  FeatureTable(const FeatureTable&) = delete;
  FeatureTable& operator=(const FeatureTable&) = delete;

  // idx and out are device-accessible USM; out holds n * dim floats.
  template <class Index>
  sycl::event gather(const Index* idx, int64_t n, float* out,
                     const std::vector<sycl::event>& deps = {}) const {
    const int64_t vecs = dim_ / 4, rpc = rows_per_chunk_;
    float* const* chunks = chunk_ptrs_;
    auto* dst = reinterpret_cast<sycl::float4*>(out);
    return q_.submit([&](sycl::handler& h) {
      h.depends_on(deps);
      h.parallel_for(sycl::range<1>(static_cast<size_t>(n * vecs)), [=](sycl::id<1> t) {
        const int64_t i = t / vecs, c = t % vecs;
        const int64_t row = static_cast<int64_t>(idx[i]);
        const auto* src = reinterpret_cast<const sycl::float4*>(chunks[row / rpc]);
        dst[t] = src[(row % rpc) * vecs + c];
      });
    });
  }

  int64_t rows() const { return rows_; }
  int64_t dim() const { return dim_; }
  size_t num_chunks() const { return chunks_.size(); }

 private:
  sycl::queue& q_;
  int64_t rows_, dim_, rows_per_chunk_;
  std::vector<float*> chunks_;
  float** chunk_ptrs_ = nullptr;
};

}  // namespace gnnsample
