// Python bindings: gnnsample._C.Sampler wraps the SYCL sampler for tests and benchmarks.
// The graph is copied once into device USM; results come back as NumPy arrays.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "gnnsample/gather.hpp"
#include "gnnsample/multi.hpp"
#include "gnnsample/sampler.hpp"

namespace nb = nanobind;
using namespace gnnsample;

template <class T>
using Arr1 = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> to_numpy(std::vector<T>&& v) {
  auto* heap = new std::vector<T>(std::move(v));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return {heap->data(), {heap->size()}, owner};
}

// Device and host copies of a graph reordering (D23) for Sampler / MultiSampler::set_reorder.
// order[new] = caller's id; pos_shift (optional) maps CSC positions back.
class ReorderStore {
 public:
  explicit ReorderStore(sycl::queue& q) : q_(q) {}
  ~ReorderStore() { clear(); }

  Reorder set(const int32_t* order, int64_t n, const int64_t* shift) {
    clear();
    perm_.assign(n, -1);
    for (int64_t i = 0; i < n; ++i) {
      const int32_t o = order[i];
      if (o < 0 || o >= n || perm_[o] != -1) throw std::invalid_argument("order is not a permutation");
      perm_[o] = static_cast<int32_t>(i);
    }
    order_ = sycl::malloc_device<int32_t>(n, q_);
    q_.memcpy(order_, order, n * sizeof(int32_t));
    if (shift) {
      shift_ = sycl::malloc_device<int64_t>(n, q_);
      q_.memcpy(shift_, shift, n * sizeof(int64_t));
    }
    q_.wait();
    return Reorder{order_, perm_.data(), shift_, n};
  }
  void clear() {
    if (order_) sycl::free(order_, q_);
    if (shift_) sycl::free(shift_, q_);
    order_ = nullptr;
    shift_ = nullptr;
    perm_.clear();
  }

 private:
  sycl::queue& q_;
  int32_t* order_ = nullptr;
  int64_t* shift_ = nullptr;
  std::vector<int32_t> perm_;
};

using OptArr1I32 = std::optional<Arr1<int32_t>>;
using OptArr1I64 = std::optional<Arr1<int64_t>>;

nb::dict to_dict(SampleResult&& r) {
  nb::dict d;
  d["node"] = to_numpy(std::move(r.node));
  d["row"] = to_numpy(std::move(r.row));
  d["col"] = to_numpy(std::move(r.col));
  d["edge"] = to_numpy(std::move(r.edge));
  d["num_sampled_nodes"] = r.num_sampled_nodes;
  d["num_sampled_edges"] = r.num_sampled_edges;
  return d;
}

class PySampler {
 public:
  PySampler(Arr1<int64_t> indptr, Arr1<int32_t> indices, std::vector<int> fanouts,
            int64_t max_batch, const std::string& device, bool profiling)
      : q_(default_context(pick(device)), pick(device), props(profiling)) {
    const int64_t n = static_cast<int64_t>(indptr.shape(0)) - 1;
    const int64_t m = static_cast<int64_t>(indices.shape(0));
    indptr_ = sycl::malloc_device<int64_t>(n + 1, q_);
    indices_ = sycl::malloc_device<int32_t>(m, q_);
    q_.memcpy(indptr_, indptr.data(), (n + 1) * sizeof(int64_t));
    q_.memcpy(indices_, indices.data(), m * sizeof(int32_t)).wait();
    n_ = n;
    s_ = std::make_unique<Sampler>(q_, indptr_, indices_, n, std::move(fanouts), max_batch);
  }

  // Loads data/prepare.py's binary CSC files straight into device memory, 64 MB at a time
  // through a pinned staging buffer, dropping each chunk from the page cache after it is
  // copied. On the WSL2 iGPU, device memory lives on the Windows side, outside the VM, so
  // a 5.9 GB edge file read through the page cache (or a NumPy copy) on top of it pushed
  // the laptop past its 15.5 GB and Windows paged the VM out (Twitter-2010, D19).
  static PySampler* from_files(const std::string& indptr_path, const std::string& indices_path,
                               std::vector<int> fanouts, int64_t max_batch,
                               const std::string& device, bool profiling) {
    auto* p = new PySampler(device, profiling);
    const int64_t n = p->load(indptr_path, reinterpret_cast<void**>(&p->indptr_), 8) - 1;
    p->load(indices_path, reinterpret_cast<void**>(&p->indices_), 4);
    p->n_ = n;
    p->s_ =std::make_unique<Sampler>(p->q_, p->indptr_, p->indices_, n, std::move(fanouts),
                                      max_batch);
    return p;
  }

  ~PySampler() {
    s_.reset();
    table_.reset();
    ro_.reset();
    sycl::free(indptr_, q_);
    sycl::free(indices_, q_);
  }

  nb::dict sample(Arr1<int32_t> seeds, uint64_t seed, bool raw) {
    std::vector<int32_t> sv(seeds.data(), seeds.data() + seeds.shape(0));
    SampleResult r;
    {
      nb::gil_scoped_release release;
      r = s_->sample(sv, seed, raw);
    }
    return to_dict(std::move(r));
  }

  // The graph this sampler holds is reorder.py's renumbering: order[new id] = original id;
  // pos_shift[new id] = original indptr[order[new id]] - indptr[new id] (needed only for
  // edge positions). None switches back to the identity.
  void set_reorder(OptArr1I32 order, OptArr1I64 pos_shift) {
    if (!order) {
      s_->set_reorder(Reorder{});
      ro_.reset();
      return;
    }
    const int64_t n = static_cast<int64_t>(order->shape(0));
    if (n != num_nodes()) throw std::invalid_argument("order must have one entry per node");
    if (pos_shift && static_cast<int64_t>(pos_shift->shape(0)) != n)
      throw std::invalid_argument("pos_shift must have one entry per node");
    s_->set_reorder(Reorder{});
    if (!ro_) ro_ = std::make_unique<ReorderStore>(q_);
    s_->set_reorder(ro_->set(order->data(), n, pos_shift ? pos_shift->data() : nullptr));
  }
  void set_hubs(int64_t hubs) { s_->set_hubs(hubs); }
  int64_t num_nodes() const { return n_; }

  // Device-resident path: sample, keep results on the device, return only the counts.
  nb::dict run(Arr1<int32_t> seeds, uint64_t seed) {
    std::vector<int32_t> sv(seeds.data(), seeds.data() + seeds.shape(0));
    {
      nb::gil_scoped_release release;
      s_->run(sv, seed);
    }
    nb::dict d;
    d["num_nodes"] = s_->num_nodes_total();
    d["num_edges"] = s_->num_edges_total();
    d["num_sampled_nodes"] = s_->num_sampled_nodes();
    d["num_sampled_edges"] = s_->num_sampled_edges();
    return d;
  }

  // Copy the last run into caller-owned device memory (PyTorch XPU tensor data_ptr()s):
  // node int64[n], edge_index int64[2, E]; optionally gather feature rows into feat
  // float32[n, dim] from the table given to set_features(). Waits for completion.
  void export_to(uintptr_t node, uintptr_t edge_index, uintptr_t feat, uintptr_t edge_pos) {
    nb::gil_scoped_release release;
    s_->export_to(reinterpret_cast<int64_t*>(node), reinterpret_cast<int64_t*>(edge_index),
                  reinterpret_cast<int64_t*>(edge_pos));
    if (feat) {
      if (!table_) throw std::runtime_error("call set_features() first");
      table_->gather(s_->out_node_ptr(), s_->num_nodes_total(), reinterpret_cast<float*>(feat));
    }
    q_.wait();
  }

  void set_features(nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> feat,
                    const std::string& usm) {
    table_.reset();
    table_ = std::make_shared<FeatureTable>(q_, feat.data(), feat.shape(0), feat.shape(1),
                                            parse_usm(usm));
  }
  // Gathers feature rows for int64 node ids already on the device (e.g. a node tensor
  // export_to() wrote) into out float32[n, dim]. Waits for completion.
  void gather(uintptr_t node, int64_t n, uintptr_t out) {
    if (!table_) throw std::runtime_error("call set_features() first");
    nb::gil_scoped_release release;
    table_->gather(reinterpret_cast<const int64_t*>(node), n, reinterpret_cast<float*>(out)).wait();
  }
  std::shared_ptr<FeatureTable> table() const { return table_; }

  void set_profile(int mode) { s_->set_profile(mode); }
  void set_options(uint32_t opts) { s_->set_options(opts); }
  std::vector<std::pair<std::string, double>> stage_times() const { return s_->stage_times(); }
  std::vector<std::tuple<std::string, double, double>> trace() const { return s_->trace(); }
  double trace_host_ns() const { return s_->trace_host_ns(); }

  std::string device_name() const { return q_.get_device().get_info<sycl::info::device::name>(); }
  int64_t max_nodes() const { return s_->max_nodes(); }
  int64_t max_edges() const { return s_->max_edges(); }

  // The device graph, for a MultiSampler that shares it (PyMulti::from_sampler).
  sycl::queue& queue() { return q_; }
  const int64_t* indptr() const { return indptr_; }
  const int32_t* indices() const { return indices_; }

 private:
  static sycl::device pick(const std::string& device) {
    if (device == "gpu") return sycl::device{sycl::gpu_selector_v};
    if (device == "cpu") return sycl::device{sycl::cpu_selector_v};
    return sycl::device{sycl::default_selector_v};
  }
  // PyTorch XPU allocates in the platform default context; using it too lets our kernels
  // read and write torch tensors directly.
  static sycl::context default_context(const sycl::device& d) {
    return d.get_platform().ext_oneapi_get_default_context();
  }
  PySampler(const std::string& device, bool profiling)
      : q_(default_context(pick(device)), pick(device), props(profiling)) {}

  // Allocates device memory for a whole binary file of `elem`-byte items and fills it in
  // chunks; returns the number of items.
  int64_t load(const std::string& path, void** dst, size_t elem) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    ::fstat(fd, &st);
    const size_t bytes = static_cast<size_t>(st.st_size);
    if (bytes % elem) throw std::runtime_error(path + ": size is not a multiple of the item size");
    char* dev = static_cast<char*>(sycl::malloc_device(bytes, q_));
    if (!dev) throw std::runtime_error("device allocation failed for " + path);
    // Plain heap staging buffer: reading straight into pinned USM host memory fails with
    // EFAULT on Docker Desktop's bind mounts (the file system cannot write into pages the
    // GPU driver mapped), although it works on the container's own file system.
    constexpr size_t kChunk = size_t{64} << 20;
    std::vector<char> stage(kChunk);
    for (size_t off = 0; off < bytes;) {
      const size_t want = std::min(kChunk, bytes - off);
      size_t got = 0;
      while (got < want) {
        const ssize_t r = ::pread(fd, stage.data() + got, want - got, static_cast<off_t>(off + got));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0)
          throw std::runtime_error("read failed for " + path + ": " +
                                   (r == 0 ? std::string("unexpected end of file")
                                           : std::string(std::strerror(errno))));
        got += static_cast<size_t>(r);
      }
      q_.memcpy(dev + off, stage.data(), want).wait();
      ::posix_fadvise(fd, static_cast<off_t>(off), static_cast<off_t>(want), POSIX_FADV_DONTNEED);
      off += want;
    }
    ::close(fd);
    *dst = dev;
    return static_cast<int64_t>(bytes / elem);
  }

  static sycl::property_list props(bool profiling) {
    if (profiling)
      return {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}};
    return {sycl::property::queue::in_order{}};
  }

  sycl::queue q_;
  int64_t* indptr_ = nullptr;
  int32_t* indices_ = nullptr;
  int64_t n_ = 0;
  std::unique_ptr<Sampler> s_;
  std::shared_ptr<FeatureTable> table_;  // shared with MultiSampler.share_features()
  std::unique_ptr<ReorderStore> ro_;
};

// Benchmark wrapper for the device feature gather (bench/gather.py).
class PyGather {
 public:
  using Feat = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

  PyGather(Feat feat, const std::string& usm, const std::string& device)
      : q_(pick_device(device), sycl::property::queue::in_order{}),
        table_(q_, feat.data(), feat.shape(0), feat.shape(1), parse_usm(usm)) {}
  ~PyGather() { release(); }

  // Wall-clock seconds per repetition for gathering rows `idx` (index already resident in
  // device USM; output preallocated), measured as submit + wait.
  std::vector<double> time(Arr1<int32_t> idx, int reps) {
    prepare(idx);
    std::vector<double> out;
    for (int r = 0; r < reps; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      table_.gather(idx_, n_, out_).wait();
      out.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return out;
  }

  nb::ndarray<nb::numpy, float, nb::ndim<2>> gather(Arr1<int32_t> idx) {
    prepare(idx);
    table_.gather(idx_, n_, out_).wait();
    auto* host = new std::vector<float>(static_cast<size_t>(n_) * table_.dim());
    q_.memcpy(host->data(), out_, host->size() * sizeof(float)).wait();
    nb::capsule owner(host, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    return {host->data(), {static_cast<size_t>(n_), static_cast<size_t>(table_.dim())}, owner};
  }

 private:
  static sycl::device pick_device(const std::string& device) {
    return device == "cpu" ? sycl::device{sycl::cpu_selector_v} : sycl::device{sycl::gpu_selector_v};
  }
  void prepare(Arr1<int32_t> idx) {
    const int64_t n = static_cast<int64_t>(idx.shape(0));
    if (n > cap_) {
      release();
      idx_ = sycl::malloc_device<int32_t>(n, q_);
      out_ = sycl::malloc_device<float>(static_cast<size_t>(n) * table_.dim(), q_);
      cap_ = n;
    }
    n_ = n;
    q_.memcpy(idx_, idx.data(), n * sizeof(int32_t)).wait();
  }
  void release() {
    if (idx_) sycl::free(idx_, q_);
    if (out_) sycl::free(out_, q_);
    idx_ = nullptr;
    out_ = nullptr;
    cap_ = 0;
  }

  sycl::queue q_;
  FeatureTable table_;
  int32_t* idx_ = nullptr;
  float* out_ = nullptr;
  int64_t n_ = 0, cap_ = 0;
};

// Prototype D22: K minibatches per launch ("batched") or one minibatch per work-group
// ("megakernel"). Results per minibatch are bit-identical to Sampler.
class PyMulti {
 public:
  PyMulti(Arr1<int64_t> indptr, Arr1<int32_t> indices, std::vector<int> fanouts, int64_t max_batch,
          int k_max, const std::string& device, bool profiling)
      : q_(PySamplerContext(device), PySamplerDevice(device), queue_props(profiling)) {
    const int64_t n = static_cast<int64_t>(indptr.shape(0)) - 1;
    const int64_t m = static_cast<int64_t>(indices.shape(0));
    indptr_ = sycl::malloc_device<int64_t>(n + 1, q_);
    indices_ = sycl::malloc_device<int32_t>(m, q_);
    q_.memcpy(indptr_, indptr.data(), (n + 1) * sizeof(int64_t));
    q_.memcpy(indices_, indices.data(), m * sizeof(int32_t)).wait();
    graph_bytes_ = (n + 1) * 8 + m * 4;
    s_ = std::make_unique<MultiSampler>(q_, indptr_, indices_, n, std::move(fanouts), max_batch,
                                        k_max);
  }
  // D25: run on another sampler's device graph instead of a second copy (Orkut: 0.94 GB).
  // Our queue uses that sampler's device and context (the platform default context, as for
  // share_features), so its USM is ours to read; nanobind keeps the sampler alive. If that
  // sampler's graph is reordered, call set_reorder() here with the same order.
  static PyMulti* from_sampler(PySampler& other, std::vector<int> fanouts, int64_t max_batch,
                               int k_max, bool profiling) {
    auto* p = new PyMulti(other.queue(), profiling);
    p->s_ = std::make_unique<MultiSampler>(p->q_, other.indptr(), other.indices(),
                                           other.num_nodes(), std::move(fanouts), max_batch,
                                           k_max);
    return p;
  }
  ~PyMulti() {
    s_.reset();
    ro_.reset();
    if (indptr_) sycl::free(indptr_, q_);  // null when the graph is shared
    if (indices_) sycl::free(indices_, q_);
  }
  // D23 switches (all output-preserving); see MultiSampler.
  void set_reorder(OptArr1I32 order, OptArr1I64 pos_shift) {
    s_->set_reorder(Reorder{});
    if (!order) {
      ro_.reset();
      return;
    }
    if (pos_shift && pos_shift->shape(0) != order->shape(0))
      throw std::invalid_argument("pos_shift must have one entry per node");
    if (!ro_) ro_ = std::make_unique<ReorderStore>(q_);
    s_->set_reorder(ro_->set(order->data(), static_cast<int64_t>(order->shape(0)),
                             pos_shift ? pos_shift->data() : nullptr));
  }
  void set_hubs(int64_t hubs, bool hints) { s_->set_hubs(hubs, hints); }
  void set_schedule(bool on) { s_->set_schedule(on); }
  void set_profile(bool on) { s_->set_profile(on); }
  std::vector<std::tuple<std::string, int, double, double>> trace() const { return s_->trace(); }
  void set_table(const std::string& kind) {
    if (kind == "auto")
      s_->set_table(MultiSampler::Table::Auto);
    else if (kind == "direct")
      s_->set_table(MultiSampler::Table::Direct);
    else if (kind == "hash")
      s_->set_table(MultiSampler::Table::Hash);
    else
      throw std::invalid_argument("table must be 'auto', 'direct' or 'hash'");
  }
  std::string table() const { return s_->table_name(); }

  // Memory accounting (D25): device bytes this object owns; the graph counts only if it is
  // our own copy. Reorder maps and the feature table are not included.
  int64_t device_bytes() const { return graph_bytes_ + s_->device_bytes(); }
  nb::dict memory() const {
    nb::dict d;
    d["graph"] = graph_bytes_;
    d["tables"] = s_->table_bytes();
    d["buffers"] = s_->buffer_bytes();
    d["total"] = device_bytes();
    d["table"] = s_->table_name();
    d["direct_bytes_per_mb"] = s_->direct_bytes_per_mb();
    d["hash_bytes_per_mb"] = s_->hash_bytes_per_mb();
    return d;
  }

  // Training path (D24): export minibatch i of the last run() into torch tensors, then
  // gather its features separately, just before it is used, so only the minibatches in
  // flight hold feature rows. Both wait for completion.
  void export_to(int i, uintptr_t node, uintptr_t edge_index, uintptr_t edge_pos) {
    nb::gil_scoped_release release;
    s_->export_to(i, reinterpret_cast<int64_t*>(node), reinterpret_cast<int64_t*>(edge_index),
                  reinterpret_cast<int64_t*>(edge_pos)).wait();
  }
  void set_features(nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> feat,
                    const std::string& usm) {
    table_.reset();
    table_ = std::make_shared<FeatureTable>(q_, feat.data(), feat.shape(0), feat.shape(1),
                                            parse_usm(usm));
  }
  // Use another sampler's feature table instead of a second copy (the gathers then run on
  // that sampler's queue; nanobind keeps it alive).
  void share_features(const PySampler& other) {
    if (!other.table()) throw std::runtime_error("that sampler has no features");
    table_ = other.table();
  }
  void gather(uintptr_t node, int64_t n, uintptr_t out) {
    if (!table_) throw std::runtime_error("call set_features() or share_features() first");
    nb::gil_scoped_release release;
    table_->gather(reinterpret_cast<const int64_t*>(node), n, reinterpret_cast<float*>(out)).wait();
  }
  int64_t max_nodes() const { return s_->max_nodes(); }
  int64_t max_edges() const { return s_->max_edges(); }
  // seeds: list of int32 arrays (one per minibatch); keys: one RNG key per minibatch.
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> run(
      std::vector<Arr1<int32_t>> seeds, std::vector<uint64_t> keys, const std::string& mode,
      int wg) {
    std::vector<std::vector<int32_t>> sv;
    for (auto& a : seeds) sv.emplace_back(a.data(), a.data() + a.shape(0));
    const auto md = mode == "megakernel" ? MultiSampler::Mode::Megakernel
                                         : MultiSampler::Mode::Batched;
    {
      nb::gil_scoped_release release;
      s_->run(sv, keys, md, wg);
    }
    std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> out;
    for (int i = 0; i < static_cast<int>(sv.size()); ++i) out.push_back(s_->counts(i));
    return out;
  }
  nb::dict result(int i, bool raw) { return to_dict(s_->result(i, raw)); }
  std::string device_name() const { return q_.get_device().get_info<sycl::info::device::name>(); }

 private:
  static sycl::device PySamplerDevice(const std::string& d) {
    return d == "cpu" ? sycl::device{sycl::cpu_selector_v} : sycl::device{sycl::gpu_selector_v};
  }
  static sycl::property_list queue_props(bool profiling) {
    if (profiling)
      return {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}};
    return {sycl::property::queue::in_order{}};
  }
  static sycl::context PySamplerContext(const std::string& d) {
    return PySamplerDevice(d).get_platform().ext_oneapi_get_default_context();
  }
  PyMulti(sycl::queue& other, bool profiling)
      : q_(other.get_context(), other.get_device(), queue_props(profiling)) {}
  sycl::queue q_;
  int64_t* indptr_ = nullptr;  // own graph copy; null when shared (from_sampler)
  int32_t* indices_ = nullptr;
  int64_t graph_bytes_ = 0;
  std::unique_ptr<MultiSampler> s_;
  std::unique_ptr<ReorderStore> ro_;
  std::shared_ptr<FeatureTable> table_;
};

NB_MODULE(_C, m) {
  nb::class_<PyMulti>(m, "MultiSampler")
      .def(nb::init<Arr1<int64_t>, Arr1<int32_t>, std::vector<int>, int64_t, int, const std::string&,
                    bool>(),
           nb::arg("indptr"), nb::arg("indices"), nb::arg("fanouts"), nb::arg("max_batch"),
           nb::arg("k_max"), nb::arg("device") = "gpu", nb::arg("profiling") = false)
      .def_static("from_sampler", &PyMulti::from_sampler, nb::arg("sampler"), nb::arg("fanouts"),
                  nb::arg("max_batch"), nb::arg("k_max"), nb::arg("profiling") = false,
                  nb::rv_policy::take_ownership, nb::keep_alive<0, 1>())
      .def("set_profile", &PyMulti::set_profile, nb::arg("on"))
      .def("trace", &PyMulti::trace)
      .def("set_table", &PyMulti::set_table, nb::arg("kind"))
      .def_prop_ro("table", &PyMulti::table)
      .def("device_bytes", &PyMulti::device_bytes)
      .def("memory", &PyMulti::memory)
      .def("run", &PyMulti::run, nb::arg("seeds"), nb::arg("keys"), nb::arg("mode") = "batched",
           nb::arg("wg") = 512)
      .def("result", &PyMulti::result, nb::arg("i"), nb::arg("raw") = false)
      .def("set_reorder", &PyMulti::set_reorder, nb::arg("order").none(),
           nb::arg("pos_shift").none() = nb::none())
      .def("set_hubs", &PyMulti::set_hubs, nb::arg("hubs"), nb::arg("hints") = true)
      .def("set_schedule", &PyMulti::set_schedule, nb::arg("on"))
      .def("export_to", &PyMulti::export_to, nb::arg("i"), nb::arg("node"), nb::arg("edge_index"),
           nb::arg("edge_pos") = 0)
      .def("set_features", &PyMulti::set_features, nb::arg("feat"), nb::arg("usm") = "host")
      .def("share_features", &PyMulti::share_features, nb::arg("sampler"), nb::keep_alive<1, 2>())
      .def("gather", &PyMulti::gather, nb::arg("node"), nb::arg("n"), nb::arg("out"))
      .def_prop_ro("max_nodes", &PyMulti::max_nodes)
      .def_prop_ro("max_edges", &PyMulti::max_edges)
      .def_prop_ro("device_name", &PyMulti::device_name);
  m.attr("PACKED_HASH") = static_cast<uint32_t>(kPackedHash);
  m.attr("FAST_SCAN") = static_cast<uint32_t>(kFastScan);
  m.attr("CLEAR_USED") = static_cast<uint32_t>(kClearUsed);
  m.attr("NO_SYNC") = static_cast<uint32_t>(kNoSync);
  m.attr("PHILOX_REUSE") = static_cast<uint32_t>(kPhiloxReuse);
  m.attr("ASSIGN_FROM_SCAN") = static_cast<uint32_t>(kAssignFromScan);
  m.attr("SLOT32") = static_cast<uint32_t>(kSlot32);
  m.attr("GEN_TAGS") = static_cast<uint32_t>(kGenTags);
  m.attr("SUBGROUP_SAMPLE") = static_cast<uint32_t>(kSubgroupSample);
  m.attr("LABEL_EARLY") = static_cast<uint32_t>(kLabelEarly);
  m.attr("SMALL_TABLE") = static_cast<uint32_t>(kSmallTable);
  m.attr("DIRECT_MAP") = static_cast<uint32_t>(kDirectMap);
  m.attr("NO_NODE_SLOT") = static_cast<uint32_t>(kNoNodeSlot);
  m.attr("HUB_HINTS") = static_cast<uint32_t>(kHubHints);
  nb::class_<PyGather>(m, "Gather")
      .def(nb::init<PyGather::Feat, const std::string&, const std::string&>(), nb::arg("feat"),
           nb::arg("usm") = "device", nb::arg("device") = "gpu")
      .def("time", &PyGather::time, nb::arg("idx"), nb::arg("reps") = 5)
      .def("gather", &PyGather::gather, nb::arg("idx"));

  nb::class_<PySampler>(m, "Sampler")
      .def_static("from_files", &PySampler::from_files, nb::arg("indptr_path"),
                  nb::arg("indices_path"), nb::arg("fanouts"), nb::arg("max_batch"),
                  nb::arg("device") = "gpu", nb::arg("profiling") = false,
                  nb::rv_policy::take_ownership)
      .def(nb::init<Arr1<int64_t>, Arr1<int32_t>, std::vector<int>, int64_t, const std::string&,
                    bool>(),
           nb::arg("indptr"), nb::arg("indices"), nb::arg("fanouts"), nb::arg("max_batch"),
           nb::arg("device") = "gpu", nb::arg("profiling") = false)
      .def("sample", &PySampler::sample, nb::arg("seeds"), nb::arg("seed") = 0,
           nb::arg("raw") = false)
      .def("set_reorder", &PySampler::set_reorder, nb::arg("order").none(),
           nb::arg("pos_shift").none() = nb::none())
      .def("set_hubs", &PySampler::set_hubs, nb::arg("hubs"))
      .def("gather", &PySampler::gather, nb::arg("node"), nb::arg("n"), nb::arg("out"))
      .def("run", &PySampler::run, nb::arg("seeds"), nb::arg("seed") = 0)
      .def("export_to", &PySampler::export_to, nb::arg("node"), nb::arg("edge_index"),
           nb::arg("feat") = 0, nb::arg("edge_pos") = 0)
      .def("set_features", &PySampler::set_features, nb::arg("feat"), nb::arg("usm") = "host")
      .def("set_profile", &PySampler::set_profile, nb::arg("mode"))
      .def("set_options", &PySampler::set_options, nb::arg("options"))
      .def("stage_times", &PySampler::stage_times)
      .def("trace", &PySampler::trace)
      .def("trace_host_ns", &PySampler::trace_host_ns)
      .def_prop_ro("device_name", &PySampler::device_name)
      .def_prop_ro("max_nodes", &PySampler::max_nodes)
      .def_prop_ro("max_edges", &PySampler::max_edges)
      .def_prop_ro("num_nodes", &PySampler::num_nodes);
}
