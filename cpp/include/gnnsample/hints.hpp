// Cache hints for neighbour-list reads (decision D23, kHubHints / MultiSampler::set_hubs).
//
// On a hub-first numbering (reorder.py "degree" or "hubsort") the ids below the hub count
// are the highest-degree nodes, and their neighbour lists are one contiguous prefix of
// `indices`. Frontiers are degree-biased (a node is sampled as a neighbour in proportion to
// its out-degree), so these rows are read over and over, by every minibatch. Reads of the
// other rows touch a line once and leave; marking them "streaming" (evict first) keeps them
// from pushing hub lines out of L1 / L3. Hub reads keep the default (cached) policy.
//
// This is Intel's LSC cache-control annotation (sycl_ext_intel_cache_controls); other
// backends drop the annotation, so the code stays portable and the result never changes.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace gnnsample {

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

using StreamingRead = decltype(syclex::properties{
    intelex::read_hint<intelex::cache_control<intelex::cache_mode::streaming,
                                              syclex::cache_level::L1, syclex::cache_level::L3>>});

// indices[pos], with the streaming hint when the row belongs to a non-hub node.
inline int32_t load_nbr(const int32_t* indices, int64_t pos, bool cold) {
  if (!cold) return indices[pos];
  syclex::annotated_ptr<int32_t, StreamingRead> p(const_cast<int32_t*>(indices));
  return p[pos];
}

}  // namespace gnnsample
