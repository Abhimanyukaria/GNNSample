// Multi-hop neighbor sampler.
//
// Produces exactly the output of python/gnnsample/reference.py:
//   - each hop expands only the nodes first discovered at the previous hop;
//   - d <= k: all neighbors; d > k: Floyd's algorithm with Philox draws;
//   - new nodes get local ids in order of their first sampled edge.
// The order is made deterministic on the GPU by recording, per hash slot, the smallest
// edge index that produced the node (atomic min); a scan over "first occurrence" flags
// then hands out local ids in edge order.
//
// Optimizations are run-time options (set_options) so each one can be A/B-tested in the
// same process against the v1 behaviour (options = 0); see bench/ablate.py and
// docs/decisions.md D12.
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "hints.hpp"
#include "philox.hpp"
#include "scan.hpp"

namespace gnnsample {

struct SampleResult {
  std::vector<int32_t> node;  // global ids; local id = position
  std::vector<int32_t> row;   // local id of sampled neighbor
  std::vector<int32_t> col;   // local id of target
  std::vector<int64_t> edge;  // global CSC position
  std::vector<int64_t> num_sampled_nodes, num_sampled_edges;
};

enum SamplerOption : uint32_t {
  kPackedHash = 1u << 0,  // key, local id and first edge in one 16-byte slot (1 cache line)
  kFastScan = 1u << 1,    // single-kernel scan for small arrays, coalesced tiles otherwise
  kClearUsed = 1u << 2,   // reset only the slots the previous minibatch used
  kNoSync = 1u << 3,      // no host round-trip per hop: launch with upper bounds, counts on device
  // The following take effect on the no-sync path:
  kPhiloxReuse = 1u << 4,    // use all 4 words of each Philox call; int32 picks
  kAssignFromScan = 1u << 5, // first occurrence from the scan (flag[e+1] != flag[e])
  kSlot32 = 1u << 6,         // 32-bit per-edge hash-slot indices
  kGenTags = 1u << 7,        // generation-tagged hash slots: no table reset per minibatch
  kSubgroupSample = 1u << 8, // one sub-group per frontier node in the sampling kernel
  kLabelEarly = 1u << 9,     // label known nodes at insert and winners at assign; relabel only repeats
  kSmallTable = 1u << 10,    // hash table of cap/2 slots (still >= worst-case nodes): half the footprint
  kDirectMap = 1u << 11,     // node id = slot (num_nodes x 8 B): no key, no probing, no claim CAS
  kNoNodeSlot = 1u << 12,    // skip node_slot writes (only kClearUsed reads them)
  kHubHints = 1u << 13,      // sub-group sampling: streaming cache hint on non-hub rows (set_hubs)
};

// Graph reordering (decision D23). The sampler runs on a renumbered copy of the graph
// (python/gnnsample/reorder.py: same rows, neighbour order kept, ids mapped) but speaks the
// caller's numbering: seeds are mapped in, node ids and CSC positions are mapped out, and
// the Philox draws are keyed on the caller's id, so the output is bit-identical to sampling
// the original graph. The arrays belong to the caller (PySampler).
struct Reorder {
  const int32_t* order = nullptr;      // device: order[new id] = caller's id
  const int32_t* perm = nullptr;       // host: perm[caller's id] = new id
  const int64_t* pos_shift = nullptr;  // device, optional: caller's position = position + pos_shift[new id]
  int64_t num_nodes = 0;

  explicit operator bool() const { return order != nullptr; }
  std::vector<int32_t> map_seeds(const std::vector<int32_t>& seeds) const {
    std::vector<int32_t> out(seeds.size());
    for (size_t i = 0; i < seeds.size(); ++i) {
      if (seeds[i] < 0 || seeds[i] >= num_nodes) throw std::out_of_range("seed id out of range");
      out[i] = perm[seeds[i]];
    }
    return out;
  }
};

// Philox node id of (new) node v: the caller's id when the graph is reordered.
inline uint32_t rng_id(const int32_t* order, int32_t v) {
  return static_cast<uint32_t>(order ? order[v] : v);
}

// Copies one sampled subgraph to the host. With a reordering (and raw = false), node ids and
// CSC positions are first mapped back to the caller's numbering on the device; without
// pos_shift, positions come back as -1. raw = true keeps the internal numbering (locality
// statistics).
inline void copy_result(sycl::queue& q, const Reorder& ro, bool raw, const int32_t* nodes,
                        const int32_t* row, const int32_t* col, const int64_t* epos, int64_t n,
                        int64_t e, SampleResult& r) {
  r.node.resize(n);
  r.row.resize(e);
  r.col.resize(e);
  r.edge.resize(e);
  q.memcpy(r.row.data(), row, e * sizeof(int32_t));
  q.memcpy(r.col.data(), col, e * sizeof(int32_t));
  if (!ro || raw) {
    q.memcpy(r.node.data(), nodes, n * sizeof(int32_t));
    q.memcpy(r.edge.data(), epos, e * sizeof(int64_t));
    q.wait();
    return;
  }
  int32_t* tn = sycl::malloc_device<int32_t>(std::max<int64_t>(n, 1), q);
  int64_t* te = sycl::malloc_device<int64_t>(std::max<int64_t>(e, 1), q);
  const int32_t* order = ro.order;
  const int64_t* shift = ro.pos_shift;
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { tn[i] = order[nodes[i]]; });
  q.parallel_for(sycl::range<1>(e), [=](sycl::id<1> i) {
    te[i] = shift ? epos[i] + shift[nodes[col[i]]] : -1;
  });
  q.memcpy(r.node.data(), tn, n * sizeof(int32_t));
  q.memcpy(r.edge.data(), te, e * sizeof(int64_t));
  q.wait();
  sycl::free(tn, q);
  sycl::free(te, q);
}

// Hash table over one allocation of cap * 16 bytes, in one of two layouts:
//   split   keys[cap] int32 | vals[cap] int32 | first[cap] int64   (v1)
//   packed  {key int32, val int32, first int64} per slot
// Accessors take strides so kernels are layout-agnostic.
struct HashView {
  int32_t* base;
  int64_t key_off, key_stride, val_off, val_stride, first_off, first_stride, mask;

  int32_t* key(int64_t s) const { return base + key_off + s * key_stride; }
  int32_t* val(int64_t s) const { return base + val_off + s * val_stride; }
  int64_t* first(int64_t s) const {
    return reinterpret_cast<int64_t*>(base) + first_off + s * first_stride;
  }

  static uint32_t mix(uint32_t x) {  // murmur3 finalizer
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
  }

  // Linear-probing insert; returns the slot holding `k` (existing or newly claimed).
  int64_t insert(int32_t k) const {
    int64_t s = mix(static_cast<uint32_t>(k)) & mask;
    while (true) {
      sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          a(*key(s));
      int32_t cur = a.load();
      if (cur == k) return s;
      if (cur == -1) {
        int32_t expected = -1;
        if (a.compare_exchange_strong(expected, k) || expected == k) return s;
      }
      s = (s + 1) & mask;
    }
  }

  void clear(int64_t s) const {
    *key(s) = -1;
    *val(s) = -1;
    *first(s) = INT64_MAX;
  }

  // Interface shared with GenHashView (used by the no-sync kernels).
  int32_t get_val(int64_t s) const { return *val(s); }
  void set_val(int64_t s, int32_t v) const { *val(s) = v; }
  int64_t get_first(int64_t s) const { return *first(s); }
  void min_first(int64_t s, int64_t e) const {
    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*first(s))
        .fetch_min(e);
  }
};

// Generation-tagged hash table (kGenTags): no reset between minibatches. 16-byte slots,
// the same size as the packed untagged table, each field tagged with the generation:
//   gk = gen << 32 | key        (64 bit) claimed with a CAS; another generation = free
//   gv = gen8 << 24 | local id  (32 bit) valid only if the 8-bit generation matches
//   gf = ~gen8 << 24 | edge     (32 bit) atomic-min target: older generations have a
//                               larger ~gen8, so current values replace them unread
// Nothing is used before it is known to belong to the current generation, so no thread
// can see a half-initialised slot. gen runs 1..255; the table is fully reset when it wraps
// (amortised over 255 minibatches). Needs local ids and per-hop edge indices < 2^24
// (Sampler checks; batch 1024 with [15,10,5] uses < 2^20).
struct GenHashView {
  static constexpr uint32_t kMaxGen = 255, kLow24 = (1u << 24) - 1;
  uint64_t* base;
  int64_t mask;
  uint32_t gen;  // 1..kMaxGen

  uint64_t* gk(int64_t s) const { return base + 2 * s; }
  uint32_t* gv(int64_t s) const { return reinterpret_cast<uint32_t*>(base + 2 * s + 1); }
  uint32_t* gf(int64_t s) const { return reinterpret_cast<uint32_t*>(base + 2 * s + 1) + 1; }

  int64_t insert(int32_t k) const {
    const uint64_t want = (uint64_t{gen} << 32) | static_cast<uint32_t>(k);
    int64_t s = HashView::mix(static_cast<uint32_t>(k)) & mask;
    while (true) {
      sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          a(*gk(s));
      uint64_t cur = a.load();
      while (true) {
        if (cur == want) return s;
        if ((cur >> 32) == gen) break;                       // taken by another key
        if (a.compare_exchange_strong(cur, want)) return s;  // stale or empty: claim it
      }  // a failed CAS reloads cur; re-check the same slot
      s = (s + 1) & mask;
    }
  }
  int32_t get_val(int64_t s) const {
    const uint32_t v = *gv(s);
    return (v >> 24) == gen ? static_cast<int32_t>(v & kLow24) : -1;
  }
  void set_val(int64_t s, int32_t v) const { *gv(s) = (gen << 24) | static_cast<uint32_t>(v); }
  int64_t get_first(int64_t s) const {
    const uint32_t f = *gf(s);
    return (f >> 24) == (~gen & 0xFFu) ? static_cast<int64_t>(f & kLow24) : INT64_MAX;
  }
  void min_first(int64_t s, int64_t e) const {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*gf(s))
        .fetch_min(((~gen & 0xFFu) << 24) | static_cast<uint32_t>(e));
  }
  void clear(int64_t s) const {  // full reset: at first use and when gen wraps
    *gk(s) = 0;
    *gv(s) = 0;
    *gf(s) = UINT32_MAX;
  }
};

// Per-hop sizes and offsets, kept on the device by the no-sync path.
struct HopState {
  int64_t lo;  // first local id of this hop's frontier
  int64_t nf;  // frontier size
  int64_t e0;  // first edge index of this hop
  int64_t ne;  // edges sampled in this hop
};

// Direct-mapped dedup table (kDirectMap): the node id is the slot, so there is no key, no
// probing and no claim CAS; only the atomic-min on the first edge remains. 8 bytes per
// graph node (not per batch), generation-tagged like GenHashView:
//   base[2k]   = gen8 << 24 | local id
//   base[2k+1] = ~gen8 << 24 | first edge   (atomic-min target)
// bench/hash_layout_stats.py: removes the 61-74% of inserts that CAS a new key, but
// touches as many 64-byte lines as the hash table (sampled ids are scattered).
struct DirectView {
  uint32_t* base;
  uint32_t gen;  // 1..GenHashView::kMaxGen

  int64_t insert(int32_t k) const { return k; }
  int32_t get_val(int64_t s) const {
    const uint32_t v = base[2 * s];
    return (v >> 24) == gen ? static_cast<int32_t>(v & GenHashView::kLow24) : -1;
  }
  void set_val(int64_t s, int32_t v) const {
    base[2 * s] = (gen << 24) | static_cast<uint32_t>(v);
  }
  int64_t get_first(int64_t s) const {
    const uint32_t f = base[2 * s + 1];
    return (f >> 24) == (~gen & 0xFFu) ? static_cast<int64_t>(f & GenHashView::kLow24)
                                       : INT64_MAX;
  }
  void min_first(int64_t s, int64_t e) const {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(base[2 * s + 1])
        .fetch_min(((~gen & 0xFFu) << 24) | static_cast<uint32_t>(e));
  }
  void clear(int64_t s) const {
    base[2 * s] = 0;
    base[2 * s + 1] = UINT32_MAX;
  }
};

class Sampler {
 public:
  static constexpr int kMaxFanout = 64;
  // Sub-group width of the cooperative sampling kernel: 16 on Intel Xe, 32 for NVIDIA.
  static constexpr int kSubGroup = 16;

  // indptr/indices must be USM pointers readable by the queue's device.
  Sampler(sycl::queue& q, const int64_t* indptr, const int32_t* indices, int64_t num_nodes,
          std::vector<int> fanouts, int64_t max_batch)
      : q_(q), indptr_(indptr), indices_(indices), num_nodes_(num_nodes),
        fanouts_(std::move(fanouts)) {
    if (!q.is_in_order()) throw std::invalid_argument("Sampler needs an in-order queue");
    for (int k : fanouts_)
      if (k < 1 || k > kMaxFanout) throw std::invalid_argument("fanout must be in [1, 64]");
    // Worst case: every sampled edge discovers a new node.
    int64_t layer = max_batch, edges = 0;
    for (int k : fanouts_) {
      layer *= k;
      edges += layer;
      b_prod_max_ = std::max(b_prod_max_, layer);
    }
    max_edges_ = edges;
    max_nodes_ = std::min<int64_t>(max_batch + edges, num_nodes_);
    cap_ = 1;
    while (cap_ < 2 * max_nodes_) cap_ <<= 1;

    nodes_ = sycl::malloc_device<int32_t>(max_nodes_, q);
    node_slot_ = sycl::malloc_device<int64_t>(max_nodes_, q);
    take_ = sycl::malloc_device<int64_t>(max_nodes_ + 1, q);
    row_ = sycl::malloc_device<int32_t>(max_edges_, q);
    col_ = sycl::malloc_device<int32_t>(max_edges_, q);
    epos_ = sycl::malloc_device<int64_t>(max_edges_, q);
    nbr_ = sycl::malloc_device<int32_t>(max_edges_, q);
    slot_ = sycl::malloc_device<int64_t>(max_edges_, q);
    flag_ = sycl::malloc_device<int64_t>(max_edges_ + 1, q);
    table_ = sycl::malloc_device<int32_t>(4 * cap_, q);
    state_ = sycl::malloc_device<HopState>(fanouts_.size() + 1, q);
    gtable_ = nullptr;  // allocated on first kGenTags use (16 bytes per slot)
    // Tagged slots keep local ids and per-hop edge indices in 24 bits.
    gen_ok_ = max_nodes_ < (int64_t{1} << 24) && b_prod_max_ < (int64_t{1} << 24);
    scan_ = std::make_unique<ScanWorkspace>(q, std::max(max_nodes_, max_edges_) + 1);
  }

  ~Sampler() {
    for (void* p : {static_cast<void*>(nodes_), static_cast<void*>(node_slot_),
                    static_cast<void*>(take_), static_cast<void*>(row_),
                    static_cast<void*>(col_), static_cast<void*>(epos_),
                    static_cast<void*>(nbr_), static_cast<void*>(slot_),
                    static_cast<void*>(flag_), static_cast<void*>(table_),
                    static_cast<void*>(state_)})
      sycl::free(p, q_);
    if (gtable_) sycl::free(gtable_, q_);
    if (dtable_) sycl::free(dtable_, q_);
    if (onodes_) sycl::free(onodes_, q_);
  }

  void set_options(uint32_t opts) { opts_ = opts; }
  uint32_t options() const { return opts_; }

  // The graph given to the constructor is a renumbering of the caller's (see Reorder);
  // pass a default Reorder{} to switch back.
  void set_reorder(const Reorder& ro) {
    ro_ = ro;
    ro_.num_nodes = num_nodes_;
    if (ro_ && !onodes_) onodes_ = sycl::malloc_device<int32_t>(max_nodes_, q_);
  }
  // kHubHints: ids below `hubs` are hub rows (hub-first numbering).
  void set_hubs(int64_t hubs) { hubs_ = hubs; }

  // Samples one minibatch; results stay in device USM (node_ptr(), ...).
  void run(const std::vector<int32_t>& seeds_in, uint64_t seed) {
    const std::vector<int32_t>& seeds = ro_ ? (mapped_ = ro_.map_seeds(seeds_in)) : seeds_in;
    if (opts_ & kNoSync) return run_nosync(seeds, seed);
    const int64_t b = static_cast<int64_t>(seeds.size());
    const bool fast_scan = opts_ & kFastScan;
    std::vector<int64_t> n_nodes{b}, n_edges;
    begin_stages();

    const HashView h = view();
    reset(h);
    mark("reset");
    q_.memcpy(nodes_, seeds.data(), b * sizeof(int32_t));
    {  // Seeds take local ids 0..b-1 (assumed unique).
      int32_t* nodes = nodes_;
      int64_t* node_slot = node_slot_;
      q_.parallel_for(sycl::range<1>(b), [=](sycl::id<1> i) {
        const int64_t s = h.insert(nodes[i]);
        *h.val(s) = static_cast<int32_t>(i);
        node_slot[i] = s;
      });
    }
    mark("seeds");

    int64_t lo = 0, hi = b, e0 = 0;
    for (size_t hop = 0; hop < fanouts_.size(); ++hop) {
      const int64_t nf = hi - lo, k = fanouts_[hop];
      const int64_t* indptr = indptr_;
      const int32_t* indices = indices_;
      const int32_t* frontier = nodes_ + lo;
      int64_t* take = take_;

      // 1. Edges per frontier node, then offsets.
      q_.parallel_for(sycl::range<1>(nf), [=](sycl::id<1> i) {
        const int32_t v = frontier[i];
        take[i] = sycl::min<int64_t>(indptr[v + 1] - indptr[v], k);
      });
      mark("count");
      scan_->exclusive_scan(take, take, nf, fast_scan);
      int64_t ne = 0;
      q_.memcpy(&ne, take + nf, sizeof(int64_t)).wait();
      mark("scan-count");

      // 2. Sample: positions, targets and neighbor ids.
      int64_t* epos = epos_ + e0;
      int32_t* col = col_ + e0;
      int32_t* nbr = nbr_ + e0;
      const int32_t lo32 = static_cast<int32_t>(lo);
      const uint32_t hop32 = static_cast<uint32_t>(hop);
      const int32_t* rng = ro_.order;
      q_.parallel_for(sycl::range<1>(nf), [=](sycl::id<1> i) {
        const int32_t v = frontier[i];
        const int64_t start = indptr[v], d = indptr[v + 1] - start, o = take[i];
        if (d <= k) {
          for (int64_t j = 0; j < d; ++j) {
            epos[o + j] = start + j;
            col[o + j] = lo32 + static_cast<int32_t>(i);
            nbr[o + j] = indices[start + j];
          }
          return;
        }
        int64_t picks[kMaxFanout];
        for (int64_t t = 0; t < k; ++t) {
          const int64_t j = d - k + t;
          int64_t p = bounded(draw_u32(seed, hop32, rng_id(rng, v), static_cast<uint32_t>(t)),
                              static_cast<uint64_t>(j + 1));
          for (int64_t u = 0; u < t; ++u)
            if (picks[u] == p) {
              p = j;
              break;
            }
          picks[t] = p;
          epos[o + t] = start + p;
          col[o + t] = lo32 + static_cast<int32_t>(i);
          nbr[o + t] = indices[start + p];
        }
      });
      mark("sample");

      // 3. Insert neighbors; record the first edge index for nodes new in this hop.
      int64_t* slot = slot_;
      int64_t* flag = flag_;
      q_.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> e) {
        const int64_t s = h.insert(nbr[e]);
        slot[e] = s;
        if (*h.val(s) < 0)
          sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(*h.first(s))
              .fetch_min(static_cast<int64_t>(e));
      });
      mark("hash-insert");
      // 4. Flag first occurrences of new nodes, scan to get their rank.
      q_.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> e) {
        const int64_t s = slot[e];
        flag[e] = *h.val(s) < 0 && *h.first(s) == static_cast<int64_t>(e);
      });
      mark("flag");
      scan_->exclusive_scan(flag, flag, ne, fast_scan);
      int64_t nnew = 0;
      q_.memcpy(&nnew, flag + ne, sizeof(int64_t)).wait();
      mark("scan-flag");
      // 5. Assign local ids hi + rank and append to the node list.
      int32_t* nodes = nodes_;
      int64_t* node_slot = node_slot_;
      const int64_t hi_ = hi;
      q_.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> e) {
        const int64_t s = slot[e];
        if (*h.val(s) < 0 && *h.first(s) == static_cast<int64_t>(e)) {
          const int64_t local = hi_ + flag[e];
          nodes[local] = nbr[e];
          node_slot[local] = s;
          *h.val(s) = static_cast<int32_t>(local);
        }
      });
      mark("assign");
      // 6. Relabel sources.
      int32_t* row = row_ + e0;
      q_.parallel_for(sycl::range<1>(ne), [=](sycl::id<1> e) { row[e] = *h.val(slot[e]); });
      mark("relabel");

      n_edges.push_back(ne);
      n_nodes.push_back(nnew);
      e0 += ne;
      lo = hi;
      hi += nnew;
    }
    q_.wait();

    num_nodes_total_ = hi;
    num_edges_total_ = e0;
    n_nodes_ = std::move(n_nodes);
    n_edges_ = std::move(n_edges);
    table_packed_ = opts_ & kPackedHash;
    table_valid_ = true;
    finish_stages();
  }

  // run() + copy of the result to host vectors (tests, reference comparison). raw = true
  // returns a reordered graph's internal ids and positions instead of the caller's.
  SampleResult sample(const std::vector<int32_t>& seeds, uint64_t seed, bool raw = false) {
    run(seeds, seed);
    SampleResult r;
    copy_result(q_, ro_, raw, nodes_, row_, col_, epos_, num_nodes_total_, num_edges_total_, r);
    r.num_sampled_nodes = n_nodes_;
    r.num_sampled_edges = n_edges_;
    return r;
  }

  // Writes the last run() into caller-owned device USM (e.g. PyTorch XPU tensors):
  // node[n] global ids, edge_index[2, E] as (neighbor, target) local ids, and optionally
  // edge_pos[E], each sampled edge's position in the CSC indices array. On a reordered
  // graph, ids and positions are the caller's (edge_pos needs Reorder::pos_shift), and
  // out_node_ptr() holds the caller's ids for the feature gather.
  sycl::event export_to(int64_t* node, int64_t* edge_index, int64_t* edge_pos = nullptr) {
    const int64_t n = num_nodes_total_, e = num_edges_total_;
    const int32_t *nodes = nodes_, *row = row_, *col = col_;
    if (ro_) {
      if (edge_pos && !ro_.pos_shift)
        throw std::invalid_argument("edge positions of a reordered graph need pos_shift");
      const int32_t* order = ro_.order;
      const int64_t* shift = ro_.pos_shift;
      const int64_t* epos = epos_;
      int32_t* onodes = onodes_;
      q_.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const int32_t o = order[nodes[i]];
        onodes[i] = o;
        node[i] = o;
      });
      if (edge_pos)
        q_.parallel_for(sycl::range<1>(e), [=](sycl::id<1> i) {
          edge_pos[i] = epos[i] + shift[nodes[col[i]]];
        });
    } else {
      q_.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { node[i] = nodes[i]; });
      if (edge_pos) q_.memcpy(edge_pos, epos_, e * sizeof(int64_t));
    }
    return q_.parallel_for(sycl::range<1>(e), [=](sycl::id<1> i) {
      edge_index[i] = row[i];
      edge_index[e + i] = col[i];
    });
  }

  // Per-stage time of the last run(). Mode 1: host wall time with a wait after every stage
  // (distorts small stages). Mode 2: GPU timestamps of barrier markers, no host waits;
  // needs a queue created with sycl::property::queue::enable_profiling.
  void set_profile(int mode) { profile_ = mode; }
  const std::vector<std::pair<std::string, double>>& stage_times() const { return stages_; }

  const int32_t* node_ptr() const { return nodes_; }
  // Caller's node ids of the last export_to() (the internal ids unless reordered).
  const int32_t* out_node_ptr() const { return ro_ ? onodes_ : nodes_; }
  int64_t num_nodes_total() const { return num_nodes_total_; }
  int64_t num_edges_total() const { return num_edges_total_; }
  const std::vector<int64_t>& num_sampled_nodes() const { return n_nodes_; }
  const std::vector<int64_t>& num_sampled_edges() const { return n_edges_; }
  int64_t max_nodes() const { return max_nodes_; }
  int64_t max_edges() const { return max_edges_; }
  int64_t hash_capacity() const { return cap_; }

 private:
  void run_nosync(const std::vector<int32_t>& seeds, uint64_t seed) {
    const bool slot32 = opts_ & kSlot32, reuse = opts_ & kPhiloxReuse;
    begin_stages();
    auto go = [&](const auto& h) {
      if (slot32 && reuse) return run_nosync_t<int32_t, true>(seeds, seed, h);
      if (slot32) return run_nosync_t<int32_t, false>(seeds, seed, h);
      if (reuse) return run_nosync_t<int64_t, true>(seeds, seed, h);
      return run_nosync_t<int64_t, false>(seeds, seed, h);
    };
    if ((opts_ & kDirectMap) && gen_ok_) {
      const DirectView d = direct_view();
      mark("reset");
      return go(d);
    }
    if ((opts_ & kGenTags) && gen_ok_) {
      const GenHashView g = gen_view();
      mark("reset");
      return go(g);
    }
    const HashView h = view();
    reset(h);
    mark("reset");
    go(h);
  }

  // Same algorithm as run(), with one host synchronisation per minibatch instead of two
  // per hop (an 8-byte readback + wait costs ~0.6 ms on the WSL2 iGPU; see
  // cpp/tools/launch_overhead.cpp). Every kernel is launched with a host-side upper bound
  // and reads the true size from state_[hop] on the device; a one-work-item kernel
  // advances the offsets between hops.
  //   SlotT        int32_t (kSlot32) or int64_t: width of the per-edge hash-slot index
  //   PhiloxReuse  one Philox call gives 4 words; use all of them (kPhiloxReuse) instead
  //                of one call per draw. Same numbers either way.
  //   View         HashView (reset every batch) or GenHashView (kGenTags, never reset)
  template <class SlotT, bool PhiloxReuse, class View>
  void run_nosync_t(const std::vector<int32_t>& seeds, uint64_t seed, const View& h) {
    const int64_t b = static_cast<int64_t>(seeds.size());
    const int hops = static_cast<int>(fanouts_.size());
    // kLabelEarly relies on the scan-based winner test, so it implies kAssignFromScan.
    const bool label_early = opts_ & kLabelEarly;
    const bool assign_from_scan = (opts_ & kAssignFromScan) || label_early;
    // node_slot_ is only read by kClearUsed resets; kNoNodeSlot skips writing it otherwise.
    const bool keep_node_slot = !(opts_ & kNoNodeSlot) || (opts_ & kClearUsed);
    q_.memcpy(nodes_, seeds.data(), b * sizeof(int32_t));
    {
      int32_t* nodes = nodes_;
      int64_t* node_slot = node_slot_;
      HopState* st0 = state_;
      // Its own kernel: an empty seed batch launches no seed work-items, and the hop
      // kernels would otherwise read an uninitialised state (GPU fault on Orkut).
      track(q_.single_task([=] { *st0 = HopState{0, b, 0, 0}; }));
      track(q_.parallel_for(sycl::range<1>(b), [=](sycl::id<1> i) {
        const int64_t s = h.insert(nodes[i]);
        h.set_val(s, static_cast<int32_t>(i));
        if (keep_node_slot) node_slot[i] = s;
      }));
    }
    mark("seeds");

    const int64_t* indptr = indptr_;
    const int32_t* indices = indices_;
    int32_t *nodes = nodes_, *nbr_all = nbr_, *col_all = col_, *row_all = row_;
    int64_t *take = take_, *epos_all = epos_, *flag = flag_, *node_slot = node_slot_;
    SlotT* slot = reinterpret_cast<SlotT*>(slot_);
    const int32_t* rng = ro_.order;  // Philox ids on a reordered graph (D23)
    const bool hints = opts_ & kHubHints;
    const int64_t hubs = hubs_;
    int64_t nf_bound = b;
    for (int hop = 0; hop < hops; ++hop) {
      const int64_t k = fanouts_[hop];
      const int64_t ne_bound = nf_bound * k;
      HopState* st = state_ + hop;
      const uint32_t hop32 = static_cast<uint32_t>(hop);

      track(q_.parallel_for(sycl::range<1>(nf_bound), [=](sycl::id<1> i) {
        if (static_cast<int64_t>(i) >= st->nf) return;
        const int32_t v = nodes[st->lo + i];
        take[i] = sycl::min<int64_t>(indptr[v + 1] - indptr[v], k);
      }));
      mark("count");
      track(scan_->exclusive_scan_dev(take, take, &st->nf, nf_bound));
      track(q_.single_task([=] { st->ne = take[st->nf]; }));
      mark("scan-count");

      if ((opts_ & kSubgroupSample) && k <= kSubGroup) {
        // One sub-group per frontier node. Lane t computes Floyd candidate t in parallel;
        // a k-step broadcast loop then applies Floyd's sequential collision rule exactly
        // (pick_t = candidate_t unless an earlier pick equals it, then j_t), so the
        // output matches the reference bit for bit. All k neighbour loads are then in
        // flight at once, and the d <= k copy is coalesced across lanes.
        constexpr size_t kWg = 8 * kSubGroup;  // 8 frontier nodes per work-group
        const size_t global = ((nf_bound * kSubGroup + kWg - 1) / kWg) * kWg;
        track(q_.parallel_for(
            sycl::nd_range<1>{global, kWg},
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSubGroup)]] {
              const auto sg = it.get_sub_group();
              const int64_t i = static_cast<int64_t>(it.get_global_id(0) / kSubGroup);
              const int64_t lane = sg.get_local_linear_id();
              if (i >= st->nf) return;  // uniform across the sub-group
              const int64_t e0 = st->e0;
              const int32_t target = static_cast<int32_t>(st->lo + i);
              const int32_t v = nodes[st->lo + i];
              const int64_t start = indptr[v], d = indptr[v + 1] - start, o = take[i] + e0;
              const bool cold = hints && v >= hubs;  // uniform across the sub-group
              if (d <= k) {
                for (int64_t j = lane; j < d; j += kSubGroup) {
                  epos_all[o + j] = start + j;
                  col_all[o + j] = target;
                  nbr_all[o + j] = load_nbr(indices, start + j, cold);
                }
                return;
              }
              const int64_t jt = d - k + lane;  // j for this lane's draw t = lane
              int64_t cand = 0;
              if (lane < k)
                cand = bounded(draw_u32(seed, hop32, rng_id(rng, v), static_cast<uint32_t>(lane)),
                               static_cast<uint64_t>(jt + 1));
              int64_t mine = -1;  // final pick of draw t = lane
              for (int64_t t = 0; t < k; ++t) {
                const int64_t c = sycl::select_from_group(sg, cand, t);
                const bool dup = sycl::any_of_group(sg, lane < t && mine == c);
                if (lane == t) mine = dup ? jt : c;
              }
              if (lane < k) {
                epos_all[o + lane] = start + mine;
                col_all[o + lane] = target;
                nbr_all[o + lane] = load_nbr(indices, start + mine, cold);
              }
            }));
        mark("sample");
      } else {
      track(q_.parallel_for(sycl::range<1>(nf_bound), [=](sycl::id<1> i) {
        if (static_cast<int64_t>(i) >= st->nf) return;
        const int64_t e0 = st->e0;
        int64_t* epos = epos_all + e0;
        int32_t* col = col_all + e0;
        int32_t* nbr = nbr_all + e0;
        const int32_t target = static_cast<int32_t>(st->lo + i);
        const int32_t v = nodes[st->lo + i];
        const int64_t start = indptr[v], d = indptr[v + 1] - start, o = take[i];
        if (d <= k) {
          for (int64_t j = 0; j < d; ++j) {
            epos[o + j] = start + j;
            col[o + j] = target;
            nbr[o + j] = indices[start + j];
          }
          return;
        }
        using Pick = std::conditional_t<PhiloxReuse, int32_t, int64_t>;
        Pick picks[kMaxFanout];
        Philox4x32 w{};
        for (int64_t t = 0; t < k; ++t) {
          const int64_t j = d - k + t;
          uint32_t r;
          if constexpr (PhiloxReuse) {
            if ((t & 3) == 0)
              w = philox4x32(rng_id(rng, v), hop32, static_cast<uint32_t>(t >> 2), 0,
                             static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
            r = w.c[t & 3];
          } else {
            r = draw_u32(seed, hop32, rng_id(rng, v), static_cast<uint32_t>(t));
          }
          Pick p = static_cast<Pick>(bounded(r, static_cast<uint64_t>(j + 1)));
          for (int64_t u = 0; u < t; ++u)
            if (picks[u] == p) {
              p = static_cast<Pick>(j);
              break;
            }
          picks[t] = p;
          epos[o + t] = start + p;
          col[o + t] = target;
          nbr[o + t] = indices[start + p];
        }
      }));
      mark("sample");
      }

      // kLabelEarly: an edge whose neighbour already has a local id (seed or earlier hop;
      // 13-24% of edges, bench/hash_stats.py) is labelled here and marked resolved
      // (slot = -1), so flag and relabel skip its table reads. New nodes in this hop have
      // no local id until assign, so for the rest val < 0 is implied.
      track(q_.parallel_for(sycl::range<1>(ne_bound), [=](sycl::id<1> e) {
        if (static_cast<int64_t>(e) >= st->ne) return;
        const int64_t s = h.insert(nbr_all[st->e0 + e]);
        const int32_t v = h.get_val(s);
        if (label_early && v >= 0) {
          row_all[st->e0 + e] = v;
          slot[e] = static_cast<SlotT>(-1);
          return;
        }
        slot[e] = static_cast<SlotT>(s);
        if (v < 0) h.min_first(s, static_cast<int64_t>(e));
      }));
      mark("hash-insert");
      track(q_.parallel_for(sycl::range<1>(ne_bound), [=](sycl::id<1> e) {
        if (static_cast<int64_t>(e) >= st->ne) return;
        const int64_t s = slot[e];
        if (label_early) {
          flag[e] = s >= 0 && h.get_first(s) == static_cast<int64_t>(e);
          return;
        }
        flag[e] = h.get_val(s) < 0 && h.get_first(s) == static_cast<int64_t>(e);
      }));
      mark("flag");
      track(scan_->exclusive_scan_dev(flag, flag, &st->ne, ne_bound));
      mark("scan-flag");
      if (assign_from_scan) {
        // After the scan, e is a first occurrence iff its rank differs from the next one;
        // only those edges touch the hash table.
        track(q_.parallel_for(sycl::range<1>(ne_bound), [=](sycl::id<1> e) {
          if (static_cast<int64_t>(e) >= st->ne || flag[e + 1] == flag[e]) return;
          const int64_t s = slot[e];
          const int64_t local = st->lo + st->nf + flag[e];
          nodes[local] = nbr_all[st->e0 + e];
          if (keep_node_slot) node_slot[local] = s;
          h.set_val(s, static_cast<int32_t>(local));
          if (label_early) row_all[st->e0 + e] = static_cast<int32_t>(local);  // winner
        }));
      } else {
        track(q_.parallel_for(sycl::range<1>(ne_bound), [=](sycl::id<1> e) {
          if (static_cast<int64_t>(e) >= st->ne) return;
          const int64_t s = slot[e];
          if (h.get_val(s) < 0 && h.get_first(s) == static_cast<int64_t>(e)) {
            const int64_t local = st->lo + st->nf + flag[e];
            nodes[local] = nbr_all[st->e0 + e];
            if (keep_node_slot) node_slot[local] = s;
            h.set_val(s, static_cast<int32_t>(local));
          }
        }));
      }
      mark("assign");
      track(q_.parallel_for(sycl::range<1>(ne_bound), [=](sycl::id<1> e) {
        if (static_cast<int64_t>(e) >= st->ne) return;
        // kLabelEarly: only repeats of nodes new in this hop (10-16% of edges) still need
        // the table; resolved edges and winners were labelled by insert / assign.
        if (label_early && (slot[e] < 0 || flag[e + 1] != flag[e])) return;
        row_all[st->e0 + e] = h.get_val(slot[e]);
      }));
      track(q_.single_task([=] {
        st[1] = HopState{st->lo + st->nf, flag[st->ne], st->e0 + st->ne, 0};
      }));
      mark("relabel");
      nf_bound = std::min(ne_bound, num_nodes_);
    }

    std::vector<HopState> hs(hops + 1);
    q_.memcpy(hs.data(), state_, hs.size() * sizeof(HopState)).wait();
    std::vector<int64_t> n_nodes{b}, n_edges;
    for (int hop = 0; hop < hops; ++hop) {
      n_edges.push_back(hs[hop].ne);
      n_nodes.push_back(hs[hop + 1].nf);
    }
    num_nodes_total_ = hs[hops].lo + hs[hops].nf;
    num_edges_total_ = hs[hops].e0;
    n_nodes_ = std::move(n_nodes);
    n_edges_ = std::move(n_edges);
    if constexpr (std::is_same_v<View, HashView>) {
      table_packed_ = opts_ & kPackedHash;
      table_valid_ = keep_node_slot;  // clear-used may only trust node_slot_ if it was written
    } else {
      table_valid_ = false;  // node_slot_ now indexes the tagged table, not table_
    }
    finish_stages();
  }

  // Next generation of the tagged table; fully resets it on first use and on wrap-around.
  GenHashView gen_view() {
    if (!gtable_) gtable_ = sycl::malloc_device<uint64_t>(2 * cap_, q_);
    const int64_t mask = active_mask();
    if (gen_ == 0 || gen_ == GenHashView::kMaxGen || mask != gen_mask_) {
      gen_mask_ = mask;
      const GenHashView g{gtable_, cap_ - 1, 0};
      q_.parallel_for(sycl::range<1>(cap_), [=](sycl::id<1> s) { g.clear(s); });
      gen_ = 0;
    }
    return GenHashView{gtable_, mask, ++gen_};
  }

  DirectView direct_view() {
    if (!dtable_) dtable_ = sycl::malloc_device<uint32_t>(2 * num_nodes_, q_);
    if (dgen_ == 0 || dgen_ == GenHashView::kMaxGen) {
      const DirectView d{dtable_, 0};
      q_.parallel_for(sycl::range<1>(num_nodes_), [=](sycl::id<1> s) { d.clear(s); });
      dgen_ = 0;
    }
    return DirectView{dtable_, ++dgen_};
  }

  // kSmallTable probes only the first cap/2 slots; cap/2 >= max_nodes_, so it never fills.
  int64_t active_mask() const { return ((opts_ & kSmallTable) ? cap_ / 2 : cap_) - 1; }

  HashView view() const {
    if (opts_ & kPackedHash)  // slot s: int32 [4s] key, [4s+1] val; int64 [2s+1] first
      return {table_, 0, 4, 1, 4, 1, 2, active_mask()};
    return {table_, 0, 1, cap_, 1, cap_, 1, active_mask()};
  }

  // Empty every slot, or (kClearUsed) only those the previous minibatch occupied, which
  // node_slot_ recorded. A first run or a layout change always resets the whole table.
  void reset(const HashView& h) {
    const bool same_layout = table_packed_ == static_cast<bool>(opts_ & kPackedHash);
    if ((opts_ & kClearUsed) && table_valid_ && same_layout) {
      const int64_t* node_slot = node_slot_;
      q_.parallel_for(sycl::range<1>(num_nodes_total_),
                      [=](sycl::id<1> i) { h.clear(node_slot[i]); });
      return;
    }
    q_.parallel_for(sycl::range<1>(cap_), [=](sycl::id<1> s) { h.clear(s); });
  }

  void begin_stages() {
    stages_.clear();
    marks_.clear();
    if (profile_ == 1) {
      q_.wait();
      t_last_ = std::chrono::steady_clock::now();
    } else if (profile_ == 2) {
      marks_.emplace_back("", q_.ext_oneapi_submit_barrier());
    } else if (profile_ == 3) {
      trace_.clear();
      q_.wait();
      t_host0_ = std::chrono::steady_clock::now();
    }
  }
  void mark(const char* name) {
    if (profile_ == 3) {  // the stage's last kernel; no barrier, so nothing is perturbed
      marks_.emplace_back(name, last_event());
      return;
    }
    if (profile_ == 2) {
      marks_.emplace_back(name, q_.ext_oneapi_submit_barrier());
      return;
    }
    if (profile_ != 1) return;
    q_.wait();
    const auto now = std::chrono::steady_clock::now();
    add_stage(name, std::chrono::duration<double>(now - t_last_).count());
    t_last_ = now;
  }
  // Mode 2: turn barrier timestamps into per-stage GPU time once the batch is done.
  // Mode 3: keep each stage's last kernel (GPU start, end) and the host wall time.
  void finish_stages() {
    if (profile_ == 3) {
      q_.wait();
      host_ns_ = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() -
                                                          t_host0_).count();
      for (auto& [name, e] : marks_)
        trace_.push_back({name,
                          static_cast<double>(e.get_profiling_info<sycl::info::event_profiling::command_start>()),
                          static_cast<double>(e.get_profiling_info<sycl::info::event_profiling::command_end>())});
      marks_.clear();
      return;
    }
    if (profile_ != 2) return;
    q_.wait();
    auto end = [](sycl::event& e) {
      return e.get_profiling_info<sycl::info::event_profiling::command_end>();
    };
    for (size_t i = 1; i < marks_.size(); ++i)
      add_stage(marks_[i].first.c_str(),
                (end(marks_[i].second) - end(marks_[i - 1].second)) * 1e-9);
  }
  // Every kernel of the no-sync path passes its event through track(), so profile mode 3
  // can read real per-kernel timestamps. (The queue's ext_oneapi_get_last_event() did not
  // carry real kernel timing on this runtime.)
  sycl::event track(sycl::event e) {
    last_ = e;
    return e;
  }
  sycl::event last_event() const { return last_; }

  void add_stage(const char* name, double dt) {
    for (auto& [k, v] : stages_)
      if (k == name) {
        v += dt;
        return;
      }
    stages_.emplace_back(name, dt);
  }

  sycl::queue& q_;
  const int64_t* indptr_;
  const int32_t* indices_;
  int64_t num_nodes_;
  std::vector<int> fanouts_;
  int64_t max_nodes_ = 0, max_edges_ = 0, cap_ = 0;
  int32_t *nodes_ = nullptr, *row_ = nullptr, *col_ = nullptr, *nbr_ = nullptr;
  int64_t *node_slot_ = nullptr, *take_ = nullptr, *epos_ = nullptr, *slot_ = nullptr,
          *flag_ = nullptr;
  int32_t* table_ = nullptr;
  HopState* state_ = nullptr;
  uint64_t* gtable_ = nullptr;
  uint32_t gen_ = 0;
  int64_t gen_mask_ = -1;  // mask the tagged table was last used with
  uint32_t* dtable_ = nullptr;  // direct-mapped table, allocated on first kDirectMap use
  uint32_t dgen_ = 0;
  bool gen_ok_ = false;
  int64_t b_prod_max_ = 0;  // largest per-hop edge bound
  std::unique_ptr<ScanWorkspace> scan_;
  uint32_t opts_ = 0;
  bool table_packed_ = false;  // layout of the slots currently in table_
  bool table_valid_ = false;   // table_ holds only the previous run's slots (else garbage)
  int64_t num_nodes_total_ = 0, num_edges_total_ = 0;
  std::vector<int64_t> n_nodes_, n_edges_;
  Reorder ro_;
  std::vector<int32_t> mapped_;  // seeds in the internal numbering
  int32_t* onodes_ = nullptr;    // caller's ids of the last export (reordered graphs)
  int64_t hubs_ = 0;
  int profile_ = 0;
  std::vector<std::pair<std::string, sycl::event>> marks_;
  std::vector<std::pair<std::string, double>> stages_;
  std::chrono::steady_clock::time_point t_last_;
  struct TraceEntry {
    std::string name;
    double start_ns, end_ns;
  };
  std::vector<TraceEntry> trace_;
  sycl::event last_;
  std::chrono::steady_clock::time_point t_host0_;
  double host_ns_ = 0;

 public:
  // Profile mode 3: per stage, the GPU start/end (ns) of its last kernel, and the host wall time
  // of the whole run() (needs a profiling queue).
  std::vector<std::tuple<std::string, double, double>> trace() const {
    std::vector<std::tuple<std::string, double, double>> out;
    for (auto& t : trace_) out.emplace_back(t.name, t.start_ns, t.end_ns);
    return out;
  }
  double trace_host_ns() const { return host_ns_; }
};

}  // namespace gnnsample
