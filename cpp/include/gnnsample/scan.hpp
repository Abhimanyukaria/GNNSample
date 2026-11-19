// Device-wide exclusive prefix sum. out must hold n+1 values; out[n] receives the total.
// `in` may alias `out`.
//
// Two implementations, selectable at run time so they can be A/B-tested in one process:
//   basic  three passes (per-group reduce, scan of group sums, per-group scan + offset);
//          each work-item owns 8 consecutive elements, so loads are strided (uncoalesced).
//   fast   n <= kSmallMax: a single work-group, one kernel (the per-hop count arrays are
//          small, so launch/sync overhead dominated); larger n: the same three passes but
//          tiles are staged through local memory with coalesced loads and stores.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <vector>

namespace gnnsample {

class ScanWorkspace {
 public:
  static constexpr size_t kGroup = 256, kPerItem = 8, kTile = kGroup * kPerItem;
  static constexpr size_t kSmallGroup = 1024, kSmallMax = kSmallGroup * 32;

  ScanWorkspace(sycl::queue& q, size_t max_n) : q_(q) {
    max_groups_ = (max_n + kTile - 1) / kTile + 1;
    partial_ = sycl::malloc_device<int64_t>(2 * max_groups_ + 1, q);
  }
  ~ScanWorkspace() { sycl::free(partial_, q_); }
  ScanWorkspace(const ScanWorkspace&) = delete;
  ScanWorkspace& operator=(const ScanWorkspace&) = delete;

  template <class T>
  sycl::event exclusive_scan(const T* in, int64_t* out, size_t n, bool fast = false,
                             const std::vector<sycl::event>& deps = {}) {
    if (!fast) return basic(in, out, n, deps);
    if (n <= kSmallMax) return single_group(in, out, n, deps);
    return tiled(in, out, n, deps);
  }

  // Same result, but the element count is read on the device from *n_dev, so no host sync
  // is needed to launch it; n_bound >= *n_dev sizes the launch (extra work-items idle).
  template <class T>
  sycl::event exclusive_scan_dev(const T* in, int64_t* out, const int64_t* n_dev,
                                 size_t n_bound) {
    const size_t groups = n_bound ? (n_bound + kTile - 1) / kTile : 1;
    int64_t* sums = partial_;
    int64_t* offs = partial_ + max_groups_;
    q_.parallel_for(sycl::nd_range<1>{groups * kGroup, kGroup}, [=](sycl::nd_item<1> it) {
      const size_t n = static_cast<size_t>(*n_dev);
      const size_t base = it.get_group(0) * kTile, lid = it.get_local_id(0);
      int64_t s = 0;
      for (size_t j = 0; j < kPerItem; ++j) {
        const size_t i = base + j * kGroup + lid;
        if (i < n) s += static_cast<int64_t>(in[i]);
      }
      s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<int64_t>());
      if (lid == 0) sums[it.get_group(0)] = s;
    });
    scan_sums(groups, sycl::event{});
    return q_.submit([&](sycl::handler& h) {
      sycl::local_accessor<int64_t, 1> tile(sycl::range<1>(kTile), h);
      h.parallel_for(sycl::nd_range<1>{groups * kGroup, kGroup}, [=](sycl::nd_item<1> it) {
        const size_t n = static_cast<size_t>(*n_dev);
        const size_t base = it.get_group(0) * kTile, lid = it.get_local_id(0);
        if (base >= n && !(it.get_group(0) == 0)) return;  // whole tile past the end
        for (size_t j = 0; j < kPerItem; ++j) {
          const size_t i = base + j * kGroup + lid;
          tile[j * kGroup + lid] = i < n ? static_cast<int64_t>(in[i]) : 0;
        }
        sycl::group_barrier(it.get_group());
        int64_t s = 0;
        for (size_t j = 0; j < kPerItem; ++j) s += tile[lid * kPerItem + j];
        int64_t run = sycl::exclusive_scan_over_group(it.get_group(), s, sycl::plus<int64_t>()) +
                      offs[it.get_group(0)];
        for (size_t j = 0; j < kPerItem; ++j) {
          const int64_t v = tile[lid * kPerItem + j];
          tile[lid * kPerItem + j] = run;
          run += v;
        }
        sycl::group_barrier(it.get_group());
        for (size_t j = 0; j < kPerItem; ++j) {
          const size_t i = base + j * kGroup + lid;
          if (i < n) out[i] = tile[j * kGroup + lid];
        }
        if (it.get_global_id(0) == 0) out[n] = offs[groups];
      });
    });
  }

 private:
  template <class T>
  sycl::event basic(const T* in, int64_t* out, size_t n, const std::vector<sycl::event>& deps) {
    const size_t groups = n ? (n + kTile - 1) / kTile : 1;
    int64_t* sums = partial_;
    int64_t* offs = partial_ + max_groups_;
    auto e1 = q_.submit([&](sycl::handler& h) {
      h.depends_on(deps);
      h.parallel_for(sycl::nd_range<1>{groups * kGroup, kGroup}, [=](sycl::nd_item<1> it) {
        const size_t base = it.get_global_id(0) * kPerItem;
        int64_t s = 0;
        for (size_t j = 0; j < kPerItem; ++j)
          if (base + j < n) s += static_cast<int64_t>(in[base + j]);
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<int64_t>());
        if (it.get_local_id(0) == 0) sums[it.get_group(0)] = s;
      });
    });
    auto e2 = scan_sums(groups, e1);
    return q_.submit([&](sycl::handler& h) {
      h.depends_on(e2);
      h.parallel_for(sycl::nd_range<1>{groups * kGroup, kGroup}, [=](sycl::nd_item<1> it) {
        const size_t base = it.get_global_id(0) * kPerItem;
        int64_t v[kPerItem];
        int64_t s = 0;
        for (size_t j = 0; j < kPerItem; ++j) {
          v[j] = base + j < n ? static_cast<int64_t>(in[base + j]) : 0;
          s += v[j];
        }
        int64_t run = sycl::exclusive_scan_over_group(it.get_group(), s, sycl::plus<int64_t>()) +
                      offs[it.get_group(0)];
        // Every work-item has read its inputs before any writes (group scan is a barrier),
        // so in-place use is safe within a group; groups touch disjoint tiles.
        for (size_t j = 0; j < kPerItem; ++j)
          if (base + j < n) {
            out[base + j] = run;
            run += v[j];
          }
        if (it.get_global_id(0) == 0) out[n] = offs[groups];
      });
    });
  }

  // One work-group; work-item i owns a contiguous chunk of ceil(n / kSmallGroup) elements.
  template <class T>
  sycl::event single_group(const T* in, int64_t* out, size_t n,
                           const std::vector<sycl::event>& deps) {
    const size_t chunk = (n + kSmallGroup - 1) / kSmallGroup;
    return q_.submit([&](sycl::handler& h) {
      h.depends_on(deps);
      h.parallel_for(sycl::nd_range<1>{kSmallGroup, kSmallGroup}, [=](sycl::nd_item<1> it) {
        const size_t lo = it.get_local_id(0) * chunk;
        const size_t hi = sycl::min(lo + chunk, n);
        int64_t s = 0;
        for (size_t j = lo; j < hi; ++j) s += static_cast<int64_t>(in[j]);
        int64_t run = sycl::exclusive_scan_over_group(it.get_group(), s, sycl::plus<int64_t>());
        const int64_t total = sycl::group_broadcast(it.get_group(), run + s, kSmallGroup - 1);
        for (size_t j = lo; j < hi; ++j) {
          const int64_t v = static_cast<int64_t>(in[j]);
          out[j] = run;
          run += v;
        }
        if (it.get_local_id(0) == 0) out[n] = total;
      });
    });
  }

  // Three passes; tiles are loaded/stored coalesced (element lid + j * kGroup) and staged
  // in local memory so each work-item can still scan 8 consecutive elements.
  template <class T>
  sycl::event tiled(const T* in, int64_t* out, size_t n, const std::vector<sycl::event>& deps) {
    const size_t groups = (n + kTile - 1) / kTile;
    int64_t* sums = partial_;
    int64_t* offs = partial_ + max_groups_;
    auto e1 = q_.submit([&](sycl::handler& h) {
      h.depends_on(deps);
      h.parallel_for(sycl::nd_range<1>{groups * kGroup, kGroup}, [=](sycl::nd_item<1> it) {
        const size_t base = it.get_group(0) * kTile, lid = it.get_local_id(0);
        int64_t s = 0;
        for (size_t j = 0; j < kPerItem; ++j) {
          const size_t i = base + j * kGroup + lid;
          if (i < n) s += static_cast<int64_t>(in[i]);
        }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<int64_t>());
        if (lid == 0) sums[it.get_group(0)] = s;
      });
    });
    auto e2 = scan_sums(groups, e1);
    return q_.submit([&](sycl::handler& h) {
      h.depends_on(e2);
      sycl::local_accessor<int64_t, 1> tile(sycl::range<1>(kTile), h);
      h.parallel_for(sycl::nd_range<1>{groups * kGroup, kGroup}, [=](sycl::nd_item<1> it) {
        const size_t base = it.get_group(0) * kTile, lid = it.get_local_id(0);
        for (size_t j = 0; j < kPerItem; ++j) {
          const size_t i = base + j * kGroup + lid;
          tile[j * kGroup + lid] = i < n ? static_cast<int64_t>(in[i]) : 0;
        }
        sycl::group_barrier(it.get_group());
        int64_t s = 0;
        for (size_t j = 0; j < kPerItem; ++j) s += tile[lid * kPerItem + j];
        int64_t run = sycl::exclusive_scan_over_group(it.get_group(), s, sycl::plus<int64_t>()) +
                      offs[it.get_group(0)];
        for (size_t j = 0; j < kPerItem; ++j) {
          const int64_t v = tile[lid * kPerItem + j];
          tile[lid * kPerItem + j] = run;
          run += v;
        }
        sycl::group_barrier(it.get_group());
        for (size_t j = 0; j < kPerItem; ++j) {
          const size_t i = base + j * kGroup + lid;
          if (i < n) out[i] = tile[j * kGroup + lid];
        }
        if (it.get_global_id(0) == 0) out[n] = offs[groups];
      });
    });
  }

  sycl::event scan_sums(size_t groups, sycl::event dep) {
    int64_t* sums = partial_;
    int64_t* offs = partial_ + max_groups_;
    return q_.submit([&](sycl::handler& h) {
      h.depends_on(dep);
      h.parallel_for(sycl::nd_range<1>{kGroup, kGroup}, [=](sycl::nd_item<1> it) {
        sycl::joint_exclusive_scan(it.get_group(), sums, sums + groups, offs,
                                   sycl::plus<int64_t>());
        if (it.get_local_id(0) == 0) offs[groups] = offs[groups - 1] + sums[groups - 1];
      });
    });
  }

  sycl::queue& q_;
  size_t max_groups_;
  int64_t* partial_;
};

}  // namespace gnnsample
