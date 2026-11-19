// Lists SYCL devices with the properties that matter for the sampler (sub-group sizes,
// local memory, USM support) and runs a tiny kernel on each GPU as a smoke test.
#include <sycl/sycl.hpp>

#include <cstdio>
#include <numeric>
#include <vector>

int main() {
  for (const auto& dev : sycl::device::get_devices()) {
    auto sg = dev.get_info<sycl::info::device::sub_group_sizes>();
    std::printf("%s | %s\n", dev.get_info<sycl::info::device::name>().c_str(),
                dev.get_platform().get_info<sycl::info::platform::name>().c_str());
    std::printf("  compute units %u, max work-group %zu, local mem %zu KB, global mem %.1f GB\n",
                dev.get_info<sycl::info::device::max_compute_units>(),
                dev.get_info<sycl::info::device::max_work_group_size>(),
                dev.get_info<sycl::info::device::local_mem_size>() / 1024,
                dev.get_info<sycl::info::device::global_mem_size>() / 1e9);
    std::printf("  sub-group sizes:");
    for (auto s : sg) std::printf(" %zu", s);
    std::printf(" | max single alloc %.2f GB",
                dev.get_info<sycl::info::device::max_mem_alloc_size>() / 1e9);
    std::printf(" | usm shared %d, int64 atomics %d\n",
                dev.has(sycl::aspect::usm_shared_allocations),
                dev.has(sycl::aspect::atomic64));

    if (!dev.is_gpu()) continue;
    sycl::queue q{dev};
    constexpr size_t n = 1 << 20;
    int* x = sycl::malloc_shared<int>(n, q);
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) { x[i] = static_cast<int>(i) & 7; }).wait();
    long sum = std::accumulate(x, x + n, 0L);
    std::printf("  smoke kernel: sum = %ld (expected %ld) -> %s\n", sum, 7L * 8 / 2 * (n / 8),
                sum == 7L * 8 / 2 * (n / 8) ? "OK" : "FAIL");
    sycl::free(x, q);
  }
}
