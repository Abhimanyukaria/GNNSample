// Cost of the primitives behind the sampler's fixed per-batch overhead (~8 ms even for a
// one-seed batch on the laptop iGPU): kernel launch, host synchronisation, small
// device-to-host readback, and the hash-table reset fill.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <vector>

using clk = std::chrono::steady_clock;

template <class F>
double median_us(F&& f, int reps = 200) {
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    auto t0 = clk::now();
    f();
    t.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main() {
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
  std::printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  int* d = sycl::malloc_device<int>(1 << 20, q);
  int64_t* big = sycl::malloc_device<int64_t>(4 << 20, q);  // 32 MB, like the hash table
  int64_t host = 0;

  auto empty = [&] { q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) { d[0] = 1; }); };
  empty();
  q.wait();
  std::printf("empty kernel + wait            %8.1f us\n", median_us([&] { empty(); q.wait(); }));
  std::printf("30 empty kernels + 1 wait      %8.1f us\n", median_us([&] {
                for (int i = 0; i < 30; ++i) empty();
                q.wait();
              }, 50));
  std::printf("8-byte device->host + wait     %8.1f us\n",
              median_us([&] { q.memcpy(&host, big, 8).wait(); }));
  std::printf("kernel over 1M items + wait    %8.1f us\n", median_us([&] {
                q.parallel_for(sycl::range<1>(1 << 20), [=](sycl::id<1> i) { d[i] = i; }).wait();
              }));
  std::printf("fill 32 MB + wait              %8.1f us\n",
              median_us([&] { q.fill(big, int64_t{-1}, 4 << 20).wait(); }, 50));
  std::printf("kernel writing 32 MB + wait    %8.1f us\n", median_us([&] {
                q.parallel_for(sycl::range<1>(4 << 20), [=](sycl::id<1> i) { big[i] = -1; }).wait();
              }, 50));
  sycl::free(d, q);
  sycl::free(big, q);

  // Memory latency vs throughput. A dependent pointer chase (one work-item, every load
  // waits for the previous one) gives pure latency; many independent random reads show how
  // much of that latency the GPU hides when it has enough work in flight.
  std::printf("\n%-34s %14s %20s\n", "random access over", "chase ns/load", "independent Gloads/s");
  for (size_t bytes : {size_t{64} << 10, size_t{4} << 20, size_t{32} << 20, size_t{256} << 20}) {
    const size_t n = bytes / sizeof(uint32_t);
    std::vector<uint32_t> perm(n), next(n);
    for (size_t i = 0; i < n; ++i) perm[i] = static_cast<uint32_t>(i);
    uint64_t r = 0x9E3779B97F4A7C15ull;
    for (size_t i = n - 1; i > 0; --i) {  // Fisher-Yates -> one random cycle
      r ^= r << 13; r ^= r >> 7; r ^= r << 17;
      std::swap(perm[i], perm[r % (i + 1)]);
    }
    for (size_t i = 0; i < n; ++i) next[perm[i]] = perm[(i + 1) % n];
    uint32_t* dn = sycl::malloc_device<uint32_t>(n, q);
    uint32_t* out = sycl::malloc_device<uint32_t>(1 << 20, q);
    q.memcpy(dn, next.data(), bytes).wait();
    constexpr int kSteps = 200000;
    auto chase = [&] {
      q.single_task([=] {
        uint32_t p = 0;
        for (int s = 0; s < kSteps; ++s) p = dn[p];
        out[0] = p;
      }).wait();
    };
    chase();
    const double chase_ns = median_us(chase, 5) * 1e3 / kSteps;
    constexpr size_t kItems = 1 << 20, kPer = 16;
    auto indep = [&] {
      q.parallel_for(sycl::range<1>(kItems), [=](sycl::id<1> i) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u, acc = 0;
        for (size_t j = 0; j < kPer; ++j) {
          h ^= h << 13; h ^= h >> 17; h ^= h << 5;
          acc += dn[h % n];  // independent: next address does not depend on the load
        }
        out[i] = acc;
      }).wait();
    };
    indep();
    const double indep_us = median_us(indep, 9);
    char label[64];
    std::snprintf(label, sizeof label, "%zu KB array", bytes >> 10);
    std::printf("%-34s %14.1f %20.2f\n", label, chase_ns, kItems * kPer / indep_us * 1e-3);
    sycl::free(dn, q);
    sycl::free(out, q);
  }
}
