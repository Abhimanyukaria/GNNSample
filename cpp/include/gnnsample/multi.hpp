// Prototype (decision D22): several minibatches per launch, two ways.
//
//   run(..., Mode::Batched)     "K per launch": every kernel of the no-sync pipeline covers K
//                               minibatches at once (padded per-minibatch segments), so the
//                               ~35 launches, the start-up and the small phases (hop 1) are
//                               shared by K minibatches; phases still spread over the whole GPU.
//   run(..., Mode::Megakernel)  "one minibatch per work-group": a single nd_range kernel in
//                               which work-group m runs minibatch m end to end; phases are
//                               separated by group barriers, never by device-wide syncs.
//
// Both reproduce, minibatch by minibatch, exactly what Sampler (and reference.py) returns for
// the same seeds and key: the same Floyd draws, local-id order (first edge within the hop)
// and edge order. They support the final configuration only: direct-mapped generation-tagged
// dedup tables (one per minibatch), label-early, sub-group sampling, fanouts <= 16, and fewer
// than 2^24 local ids / per-hop edges.
//
// Decision D23 adds, all off by default and all output-preserving:
//   set_reorder()   run on a renumbered graph, speak the caller's ids (see Reorder);
//   set_hubs(h)     streaming cache hint on reads of non-hub rows (ids >= h; hints.hpp);
//   set_schedule()  batched mode only: cross-minibatch locality schedule. Each hop's count
//                   and sampling kernels visit the K frontiers' nodes grouped by node-id
//                   bucket instead of minibatch by minibatch, so a node (and its id
//                   neighbourhood) that several minibatches expand is read by neighbouring
//                   sub-groups at about the same time and its lines are fetched once.
//                   Every node still writes to its own minibatch's slots, so the output is
//                   unchanged.
//
// Decision D25 (memory), output-preserving:
//   set_table()     dedup table per minibatch: "direct" (8 bytes per graph node, as above),
//                   "hash" (GenHashView, 16 bytes per slot, capacity = the smallest power of
//                   two >= max_nodes(): the active half of Sampler's kSmallTable table) or
//                   "auto" (default): whichever needs fewer bytes per minibatch. Direct grows
//                   with the graph (Twitter, K=8: 2.7 GB), hash with the batch (16 MB each).
//   device_bytes()  bytes this object allocated (the graph belongs to the caller).
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "sampler.hpp"

namespace gnnsample {

class MultiSampler {
 public:
  enum class Mode { Batched, Megakernel };
  enum class Table { Auto, Direct, Hash };
  static constexpr int kSG = Sampler::kSubGroup;
  static constexpr size_t kGroup = 256, kPer = 8, kTile = kGroup * kPer;
  static constexpr int64_t kBuckets = int64_t{1} << 16;  // locality schedule: node-id buckets

  MultiSampler(sycl::queue& q, const int64_t* indptr, const int32_t* indices, int64_t num_nodes,
               std::vector<int> fanouts, int64_t max_batch, int k_max)
      : q_(q), indptr_(indptr), indices_(indices), n_(num_nodes), fan_(std::move(fanouts)),
        b_max_(max_batch), kmax_(k_max) {
    if (!q.is_in_order()) throw std::invalid_argument("MultiSampler needs an in-order queue");
    for (int k : fan_)
      if (k < 1 || k > kSG) throw std::invalid_argument("prototype supports fanouts 1..16");
    int64_t layer = max_batch, edges = 0;
    for (int k : fan_) {
      layer *= k;
      edges += layer;
      hop_e_max_ = std::max(hop_e_max_, layer);
    }
    me_ = edges;
    mn_ = std::min<int64_t>(max_batch + edges, n_);
    if (mn_ >= (int64_t{1} << 24) || hop_e_max_ >= (int64_t{1} << 24))
      throw std::invalid_argument("prototype needs < 2^24 local ids and per-hop edges");
    hops_ = static_cast<int>(fan_.size());
    sn_ = mn_ + 1;
    se_ = me_ + 1;
    // A minibatch inserts at most mn_ distinct keys per generation (each one becomes a local
    // id), so a table of hcap_ >= mn_ slots always has a free or stale slot for a new key:
    // linear probing terminates and the table can never fill.
    hcap_ = 1;
    while (hcap_ < mn_) hcap_ <<= 1;
    const size_t K = kmax_;
    nodes_ = sycl::malloc_device<int32_t>(K * sn_, q);
    take_ = sycl::malloc_device<int64_t>(K * sn_, q);
    epos_ = sycl::malloc_device<int64_t>(K * se_, q);
    col_ = sycl::malloc_device<int32_t>(K * se_, q);
    nbr_ = sycl::malloc_device<int32_t>(K * se_, q);
    row_ = sycl::malloc_device<int32_t>(K * se_, q);
    slot_ = sycl::malloc_device<int32_t>(K * se_, q);
    flag_ = sycl::malloc_device<int64_t>(K * se_, q);
    state_ = sycl::malloc_device<HopState>(K * (hops_ + 1), q);
    keys_ = sycl::malloc_device<uint64_t>(K, q);
    bsz_ = sycl::malloc_device<int32_t>(K, q);
    seeds_ = sycl::malloc_device<int32_t>(K * b_max_, q);
    const size_t tps_max = (std::max<size_t>(sn_, se_) + kTile - 1) / kTile;
    partial_ = sycl::malloc_device<int64_t>(2 * K * (tps_max + 1), q);
    pstride_ = tps_max + 1;
    for (void* p : {static_cast<void*>(nodes_), static_cast<void*>(take_),
                    static_cast<void*>(epos_), static_cast<void*>(col_), static_cast<void*>(nbr_),
                    static_cast<void*>(row_), static_cast<void*>(slot_),
                    static_cast<void*>(flag_), static_cast<void*>(state_),
                    static_cast<void*>(partial_)})
      if (!p) throw std::runtime_error("MultiSampler: device allocation failed (lower k_max)");
    set_table(Table::Auto);
  }

  ~MultiSampler() {
    for (void* p : {static_cast<void*>(nodes_), static_cast<void*>(take_),
                    static_cast<void*>(epos_), static_cast<void*>(col_), static_cast<void*>(nbr_),
                    static_cast<void*>(row_), static_cast<void*>(slot_),
                    static_cast<void*>(flag_), static_cast<void*>(dtable_),
                    static_cast<void*>(htable_),
                    static_cast<void*>(state_), static_cast<void*>(keys_),
                    static_cast<void*>(bsz_), static_cast<void*>(seeds_),
                    static_cast<void*>(partial_), static_cast<void*>(work_),
                    static_cast<void*>(bcnt_), static_cast<void*>(bcur_)})
      if (p) sycl::free(p, q_);
  }

  void set_reorder(const Reorder& ro) {
    ro_ = ro;
    ro_.num_nodes = n_;
  }
  void set_hubs(int64_t hubs, bool hints) {
    hubs_ = hubs;
    hints_ = hints;
  }
  void set_schedule(bool on) {
    if (on && kmax_ > 255) throw std::invalid_argument("locality schedule supports k_max <= 255");
    sched_ = on;
    if (on && !work_) {
      work_ = sycl::malloc_device<uint32_t>(static_cast<size_t>(kmax_) * sn_, q_);
      bcnt_ = sycl::malloc_device<uint32_t>(kBuckets, q_);
      bcur_ = sycl::malloc_device<uint32_t>(kBuckets + 1, q_);
      if (!work_ || !bcnt_ || !bcur_) throw std::runtime_error("MultiSampler: schedule allocation failed");
      while (((n_ - 1) >> bshift_) >= kBuckets) ++bshift_;
    }
  }

  // Dedup table kind (D25). Only the chosen kind stays allocated: switching frees the other,
  // and the new table starts from a full reset (gen_ = 0) at the next run().
  // Auto: hash only when the direct table is at least 1.5x larger. Measured (K=8, paired):
  // products 1.17x larger -> direct 7% faster (no probing); LiveJournal 2.3x -> hash 23%
  // faster (its entries pack into fewer lines); Orkut 1.46x -> tie; Twitter 20x -> only hash fits.
  void set_table(Table t) {
    const bool hash = t == Table::Hash ||
                      (t == Table::Auto && 3 * hash_bytes_per_mb() <= 2 * direct_bytes_per_mb());
    if (hash ? htable_ != nullptr : dtable_ != nullptr) return;
    q_.wait();  // the previous run's kernels may still use the table we free
    if (dtable_) sycl::free(dtable_, q_);
    if (htable_) sycl::free(htable_, q_);
    dtable_ = nullptr;
    htable_ = nullptr;
    const size_t K = kmax_;
    if (hash)
      htable_ = sycl::malloc_device<uint64_t>(K * 2 * static_cast<size_t>(hcap_), q_);
    else
      dtable_ = sycl::malloc_device<uint32_t>(K * 2 * static_cast<size_t>(n_), q_);
    if (!htable_ && !dtable_)
      throw std::runtime_error("MultiSampler: dedup table allocation failed (lower k_max)");
    gen_ = 0;
  }
  const char* table_name() const { return htable_ ? "hash" : "direct"; }
  int64_t hash_capacity() const { return hcap_; }  // slots per minibatch
  int64_t direct_bytes_per_mb() const { return 8 * n_; }
  int64_t hash_bytes_per_mb() const { return 16 * hcap_; }

  // Memory accounting (D25): bytes of every device allocation this object owns.
  int64_t table_bytes() const {
    return static_cast<int64_t>(kmax_) * (htable_ ? hash_bytes_per_mb() : direct_bytes_per_mb());
  }
  int64_t buffer_bytes() const {  // per-minibatch buffers, K of each
    const int64_t K = kmax_;
    int64_t b = K * sn_ * (4 + 8)              // nodes, take
                + K * se_ * (8 + 4 * 4 + 8)    // epos, col / nbr / row / slot, flag
                + K * (hops_ + 1) * static_cast<int64_t>(sizeof(HopState)) + K * (8 + 4) +
                K * b_max_ * 4                 // state, keys + bsz, seeds
                + 2 * K * static_cast<int64_t>(pstride_) * 8;  // scan partials
    if (work_) b += K * sn_ * 4 + (2 * kBuckets + 1) * 4;     // locality schedule
    return b;
  }
  int64_t device_bytes() const { return table_bytes() + buffer_bytes(); }

  // Samples K = seeds.size() minibatches; minibatch m uses RNG key keys[m]. Results stay on
  // the device; counts() / result(m) read them. wg is the megakernel's work-group size.
  void run(const std::vector<std::vector<int32_t>>& seeds, const std::vector<uint64_t>& keys,
           Mode mode, int wg = 512) {
    const int K = static_cast<int>(seeds.size());
    if (K < 1 || K > kmax_ || keys.size() != seeds.size())
      throw std::invalid_argument("need 1..k_max seed lists and one key per list");
    k_ = K;
    std::vector<int32_t> flat(static_cast<size_t>(K) * b_max_, 0), bs(K);
    for (int m = 0; m < K; ++m) {
      if (static_cast<int64_t>(seeds[m].size()) > b_max_) throw std::invalid_argument("batch too large");
      bs[m] = static_cast<int32_t>(seeds[m].size());
      const std::vector<int32_t> mapped = ro_ ? ro_.map_seeds(seeds[m]) : seeds[m];
      std::copy(mapped.begin(), mapped.end(), flat.begin() + static_cast<size_t>(m) * b_max_);
    }
    recs_.clear();
    cur_hop_ = -1;
    rec("h2d", q_.memcpy(seeds_, flat.data(), flat.size() * sizeof(int32_t)));
    rec("h2d", q_.memcpy(bsz_, bs.data(), K * sizeof(int32_t)));
    rec("h2d", q_.memcpy(keys_, keys.data(), K * sizeof(uint64_t)));
    next_generation();
    auto go = [&](const auto& tabs) {
      init_seeds(K, tabs);
      if (mode == Mode::Batched)
        run_batched(K, tabs);
      else
        run_megakernel(K, wg, tabs);
    };
    if (htable_)
      go(HashTables{htable_, hcap_, gen_});
    else
      go(DirectTables{dtable_, n_, gen_});
    hs_.resize(static_cast<size_t>(K) * (hops_ + 1));
    cur_hop_ = -1;
    rec("d2h", q_.memcpy(hs_.data(), state_, hs_.size() * sizeof(HopState))).wait();
  }

  // Per-kernel GPU timestamps (decision D25): off unless set_profile(true) on a queue
  // created with enable_profiling. trace() lists every command of the last run() as
  // (stage, hop, start ns, end ns); hop -1 = outside the hop loop.
  void set_profile(bool on) { prof_ = on; }
  std::vector<std::tuple<std::string, int, double, double>> trace() const {
    std::vector<std::tuple<std::string, int, double, double>> out;
    for (const auto& r : recs_)
      out.emplace_back(
          r.name, r.hop,
          static_cast<double>(r.e.get_profiling_info<sycl::info::event_profiling::command_start>()),
          static_cast<double>(r.e.get_profiling_info<sycl::info::event_profiling::command_end>()));
    return out;
  }

  // Per minibatch: (num_sampled_nodes per hop incl. seeds, num_sampled_edges per hop).
  std::pair<std::vector<int64_t>, std::vector<int64_t>> counts(int m) const {
    std::vector<int64_t> nn{hs_[m * (hops_ + 1)].nf}, ne;
    for (int h = 0; h < hops_; ++h) {
      ne.push_back(hs_[m * (hops_ + 1) + h].ne);
      nn.push_back(hs_[m * (hops_ + 1) + h + 1].nf);
    }
    return {nn, ne};
  }

  // Writes minibatch m of the last run() into caller-owned device USM, like
  // Sampler::export_to: node int64[n] (caller's ids), edge_index int64[2, E] as (neighbor,
  // target) local ids, optionally edge_pos int64[E] (needs pos_shift on a reordered graph).
  sycl::event export_to(int m, int64_t* node, int64_t* edge_index, int64_t* edge_pos = nullptr) {
    const HopState& last = hs_[m * (hops_ + 1) + hops_];
    const int64_t n = last.lo + last.nf, e = last.e0;
    const int32_t *nodes = nodes_ + m * sn_, *row = row_ + m * se_, *col = col_ + m * se_;
    const int64_t* epos = epos_ + m * se_;
    const int32_t* order = ro_.order;
    const int64_t* shift = ro_.pos_shift;
    if (edge_pos && order && !shift)
      throw std::invalid_argument("edge positions of a reordered graph need pos_shift");
    q_.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      node[i] = order ? order[nodes[i]] : nodes[i];
    });
    if (edge_pos)
      q_.parallel_for(sycl::range<1>(e), [=](sycl::id<1> i) {
        edge_pos[i] = shift ? epos[i] + shift[nodes[col[i]]] : epos[i];
      });
    return q_.parallel_for(sycl::range<1>(e), [=](sycl::id<1> i) {
      edge_index[i] = row[i];
      edge_index[e + i] = col[i];
    });
  }
  int64_t max_nodes() const { return mn_; }  // per minibatch
  int64_t max_edges() const { return me_; }

  // raw = true keeps a reordered graph's internal ids and positions.
  SampleResult result(int m, bool raw = false) {
    const HopState& last = hs_[m * (hops_ + 1) + hops_];
    const int64_t n = last.lo + last.nf, e = last.e0;
    SampleResult r;
    copy_result(q_, ro_, raw, nodes_ + m * sn_, row_ + m * se_, col_ + m * se_,
                epos_ + m * se_, n, e, r);
    std::tie(r.num_sampled_nodes, r.num_sampled_edges) = counts(m);
    return r;
  }

 private:
  // The K per-minibatch tables of one generation, as seen by a kernel: tabs(m) is minibatch
  // m's view. The batched and megakernel kernels are templated over these two.
  struct DirectTables {
    uint32_t* base;
    int64_t n;
    uint32_t gen;
    DirectView operator()(int64_t m) const { return DirectView{base + 2 * m * n, gen}; }
  };
  struct HashTables {
    uint64_t* base;
    int64_t cap;
    uint32_t gen;
    GenHashView operator()(int64_t m) const {
      return GenHashView{base + 2 * m * cap, cap - 1, gen};
    }
  };

  // One generation-tagged table per minibatch; all K (also those unused by a run with fewer
  // than k_max minibatches) share gen_ and advance together. Every table is fully reset at
  // first use (gen_ == 0, also after set_table) and when the 8-bit generation wraps, so a
  // tag can only match an entry written in the current generation.
  void next_generation() {
    if (gen_ == 0 || gen_ == GenHashView::kMaxGen) {
      if (htable_) {
        const GenHashView t{htable_, 0, 0};
        rec("reset", q_.parallel_for(sycl::range<1>(static_cast<int64_t>(kmax_) * hcap_),
                                     [=](sycl::id<1> s) { t.clear(s); }));
      } else {
        const DirectView t{dtable_, 0};
        rec("reset", q_.parallel_for(sycl::range<1>(static_cast<int64_t>(kmax_) * n_),
                                     [=](sycl::id<1> s) { t.clear(s); }));
      }
      gen_ = 0;
    }
    ++gen_;
  }

  // Seed i of minibatch m gets local id i. DirectView's insert is the identity (slot = node
  // id); the hash table claims a slot first.
  template <class Tabs>
  void init_seeds(int K, const Tabs& tabs) {
    const int64_t B = b_max_, sn = sn_, H = hops_;
    int32_t *nodes = nodes_, *seeds = seeds_, *bsz = bsz_;
    HopState* st = state_;
    rec("seeds", q_.parallel_for(sycl::range<1>(K * B), [=](sycl::id<1> g) {
      const int64_t m = g / B, i = g % B;
      if (i == 0) st[m * (H + 1)] = HopState{0, bsz[m], 0, 0};
      if (i >= bsz[m]) return;
      const int32_t v = seeds[m * B + i];
      nodes[m * sn + i] = v;
      const auto h = tabs(m);
      h.set_val(h.insert(v), static_cast<int32_t>(i));
    }));
  }

  // ---------------- Mode::Batched: every kernel covers all K minibatches -----------------
  // Exclusive scan of each minibatch's segment data[m*stride .. +n_m) (n_m read from the hop
  // state: nf or ne), total written to data[m*stride + n_m]. Tiles never cross segments.
  void seg_scan(const char* name, int64_t* data, int64_t stride, const HopState* st, bool use_ne,
                int64_t bound, int K) {
    const size_t tps = std::max<size_t>(1, (bound + kTile - 1) / kTile);
    const size_t ps = pstride_;
    int64_t* sums = partial_;
    int64_t* offs = partial_ + static_cast<size_t>(kmax_) * ps;
    const int64_t H1 = hops_ + 1;
    auto n_of = [=](int64_t m) { return use_ne ? st[m * H1].ne : st[m * H1].nf; };
    rec(name, q_.parallel_for(sycl::nd_range<1>{K * tps * kGroup, kGroup}, [=](sycl::nd_item<1> it) {
      const size_t grp = it.get_group(0), m = grp / tps, tile = grp % tps, lid = it.get_local_id(0);
      const int64_t n = n_of(m);
      const int64_t* in = data + m * stride;
      int64_t s = 0;
      for (size_t j = 0; j < kPer; ++j) {
        const int64_t i = static_cast<int64_t>(tile * kTile + j * kGroup + lid);
        if (i < n) s += in[i];
      }
      s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<int64_t>());
      if (lid == 0) sums[m * ps + tile] = s;
    }));
    rec(name, q_.parallel_for(sycl::nd_range<1>{K * kGroup, kGroup}, [=](sycl::nd_item<1> it) {
      const size_t m = it.get_group(0);
      sycl::joint_exclusive_scan(it.get_group(), sums + m * ps, sums + m * ps + tps, offs + m * ps,
                                 sycl::plus<int64_t>());
      if (it.get_local_id(0) == 0)
        offs[m * ps + tps] = offs[m * ps + tps - 1] + sums[m * ps + tps - 1];
    }));
    rec(name, q_.submit([&](sycl::handler& h) {
      sycl::local_accessor<int64_t, 1> tile_mem(sycl::range<1>(kTile), h);
      h.parallel_for(sycl::nd_range<1>{K * tps * kGroup, kGroup}, [=](sycl::nd_item<1> it) {
        const size_t grp = it.get_group(0), m = grp / tps, tile = grp % tps, lid = it.get_local_id(0);
        const int64_t n = n_of(m);
        int64_t* io = data + m * stride;
        const int64_t base = static_cast<int64_t>(tile * kTile);
        if (base >= n && tile != 0) return;  // whole tile past this segment's end
        for (size_t j = 0; j < kPer; ++j) {
          const int64_t i = base + static_cast<int64_t>(j * kGroup + lid);
          tile_mem[j * kGroup + lid] = i < n ? io[i] : 0;
        }
        sycl::group_barrier(it.get_group());
        int64_t s = 0;
        for (size_t j = 0; j < kPer; ++j) s += tile_mem[lid * kPer + j];
        int64_t run = sycl::exclusive_scan_over_group(it.get_group(), s, sycl::plus<int64_t>()) +
                      offs[m * ps + tile];
        for (size_t j = 0; j < kPer; ++j) {
          const int64_t v = tile_mem[lid * kPer + j];
          tile_mem[lid * kPer + j] = run;
          run += v;
        }
        sycl::group_barrier(it.get_group());
        for (size_t j = 0; j < kPer; ++j) {
          const int64_t i = base + static_cast<int64_t>(j * kGroup + lid);
          if (i < n) io[i] = tile_mem[j * kGroup + lid];
        }
        if (tile == 0 && lid == 0) io[n] = offs[m * ps + tps];
      });
    }));
  }

  template <class Tabs>
  void run_batched(int K, const Tabs& tabs) {
    const int64_t* indptr = indptr_;
    const int32_t* indices = indices_;
    int32_t *nodes = nodes_, *nbr = nbr_, *col = col_, *row = row_, *slot = slot_;
    int64_t *take = take_, *epos = epos_, *flag = flag_;
    uint64_t* keys = keys_;
    const int64_t sn = sn_, se = se_, H1 = hops_ + 1;
    const int32_t* rng = ro_.order;
    const bool hints = hints_;
    const int64_t hubs = hubs_;
    int64_t nf_bound = b_max_;
    for (int hop = 0; hop < hops_; ++hop) {
      const int64_t k = fan_[hop], ne_bound = nf_bound * k, NB = nf_bound, EB = ne_bound;
      HopState* st = state_ + hop;  // minibatch m's state for this hop: st[m * H1]
      const uint32_t hop32 = static_cast<uint32_t>(hop);
      cur_hop_ = hop;
      if (sched_) build_worklist(K, st, NB);
      const Items items{sched_ ? work_ : nullptr, bcur_ + kBuckets, st, H1, NB, K};
      rec("count", q_.parallel_for(sycl::range<1>(K * NB), [=](sycl::id<1> g) {
        int64_t m, i;
        if (!items.get(g, m, i)) return;
        const int32_t v = nodes[m * sn + st[m * H1].lo + i];
        take[m * sn + i] = sycl::min<int64_t>(indptr[v + 1] - indptr[v], k);
      }));
      seg_scan("scan-count", take, sn, st, false, NB, K);
      rec("hop-state", q_.parallel_for(sycl::range<1>(K), [=](sycl::id<1> m) {
        st[m * H1].ne = take[m * sn + st[m * H1].nf];
      }));
      constexpr size_t kWg = 8 * kSG;
      const size_t global = ((K * NB * kSG + kWg - 1) / kWg) * kWg;
      rec("sample", q_.parallel_for(sycl::nd_range<1>{global, kWg},
                      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
        const auto sg = it.get_sub_group();
        const int64_t gnode = static_cast<int64_t>(it.get_global_id(0) / kSG);
        int64_t m, i;
        if (!items.get(gnode, m, i)) return;  // uniform across the sub-group
        const HopState s = st[m * H1];
        const int32_t v = nodes[m * sn + s.lo + i];
        sample_node(sg, v, rng_id(rng, v), hints && v >= hubs, static_cast<int32_t>(s.lo + i),
                    take[m * sn + i], k, keys[m], hop32, indptr, indices,
                    epos + m * se + s.e0, col + m * se + s.e0, nbr + m * se + s.e0);
      }));
      rec("insert", q_.parallel_for(sycl::range<1>(K * EB), [=](sycl::id<1> g) {
        const int64_t m = g / EB, e = g % EB;
        const HopState s = st[m * H1];
        if (e >= s.ne) return;
        insert_edge(tabs(m), e, nbr[m * se + s.e0 + e],
                    row + m * se + s.e0, slot + m * se);
      }));
      rec("flag", q_.parallel_for(sycl::range<1>(K * EB), [=](sycl::id<1> g) {
        const int64_t m = g / EB, e = g % EB;
        if (e >= st[m * H1].ne) return;
        const int32_t sl = slot[m * se + e];
        flag[m * se + e] = sl >= 0 && tabs(m).get_first(sl) == e;
      }));
      seg_scan("scan-flag", flag, se, st, true, EB, K);
      rec("assign", q_.parallel_for(sycl::range<1>(K * EB), [=](sycl::id<1> g) {
        const int64_t m = g / EB, e = g % EB;
        const HopState s = st[m * H1];
        if (e >= s.ne) return;
        assign_or_relabel(tabs(m), e, s, flag + m * se, slot + m * se, nbr + m * se + s.e0,
                          nodes + m * sn, row + m * se + s.e0, /*assign=*/true);
      }));
      rec("relabel", q_.parallel_for(sycl::range<1>(K * EB), [=](sycl::id<1> g) {
        const int64_t m = g / EB, e = g % EB;
        const HopState s = st[m * H1];
        if (e >= s.ne) return;
        assign_or_relabel(tabs(m), e, s, flag + m * se, slot + m * se, nbr + m * se + s.e0,
                          nodes + m * sn, row + m * se + s.e0, /*assign=*/false);
      }));
      rec("hop-state", q_.parallel_for(sycl::range<1>(K), [=](sycl::id<1> m) {
        const HopState s = st[m * H1];
        st[m * H1 + 1] = HopState{s.lo + s.nf, flag[m * se + s.ne], s.e0 + s.ne, 0};
      }));
      nf_bound = std::min(ne_bound, n_);
    }
  }

  // ---------------- Mode::Megakernel: work-group m runs minibatch m --------------------
  template <class Tabs>
  void run_megakernel(int K, int wg, const Tabs& tabs) {
    if (wg % kSG || wg < kSG) throw std::invalid_argument("work-group size must be a multiple of 16");
    const int64_t* indptr = indptr_;
    const int32_t* indices = indices_;
    int32_t *nodes_all = nodes_, *nbr_all = nbr_, *col_all = col_, *row_all = row_,
            *slot_all = slot_;
    int64_t *take_all = take_, *epos_all = epos_, *flag_all = flag_;
    uint64_t* keys = keys_;
    HopState* st_all = state_;
    const int64_t sn = sn_, se = se_, H = hops_;
    const int32_t* rng = ro_.order;
    const bool hints = hints_;
    const int64_t hubs = hubs_;
    int fan[8] = {0};
    for (int h = 0; h < hops_; ++h) fan[h] = fan_[h];
    if (hops_ > 8) throw std::invalid_argument("prototype supports up to 8 hops");
    const size_t WG = static_cast<size_t>(wg);
    rec("megakernel", q_.parallel_for(sycl::nd_range<1>{K * WG, WG}, [=](sycl::nd_item<1> it)
                                                         [[sycl::reqd_sub_group_size(kSG)]] {
      const auto grp = it.get_group();
      const auto sg = it.get_sub_group();
      const int64_t m = it.get_group(0), lid = it.get_local_id(0);
      const int64_t n_sg = WG / kSG, sg_id = lid / kSG;
      int32_t* nodes = nodes_all + m * sn;
      int64_t* take = take_all + m * sn;
      int64_t* flag = flag_all + m * se;
      int32_t* slot = slot_all + m * se;
      HopState* st = st_all + m * (H + 1);
      const auto tab = tabs(m);
      const uint64_t key = keys[m];
      HopState s = st[0];  // every work-item keeps the same copy; updated in lockstep
      for (int hop = 0; hop < H; ++hop) {
        const int64_t k = fan[hop];
        for (int64_t i = lid; i < s.nf; i += WG) {
          const int32_t v = nodes[s.lo + i];
          take[i] = sycl::min<int64_t>(indptr[v + 1] - indptr[v], k);
        }
        sycl::group_barrier(grp);
        s.ne = group_scan(it, take, s.nf);
        for (int64_t i = sg_id; i < s.nf; i += n_sg) {  // one sub-group per frontier node
          const int32_t v = nodes[s.lo + i];
          sample_node(sg, v, rng_id(rng, v), hints && v >= hubs, static_cast<int32_t>(s.lo + i),
                      take[i], k, key, static_cast<uint32_t>(hop), indptr, indices,
                      epos_all + m * se + s.e0, col_all + m * se + s.e0, nbr_all + m * se + s.e0);
        }
        sycl::group_barrier(grp);
        for (int64_t e = lid; e < s.ne; e += WG)
          insert_edge(tab, e, nbr_all[m * se + s.e0 + e], row_all + m * se + s.e0, slot);
        sycl::group_barrier(grp);
        for (int64_t e = lid; e < s.ne; e += WG) {
          const int32_t sl = slot[e];
          flag[e] = sl >= 0 && tab.get_first(sl) == e;
        }
        sycl::group_barrier(grp);
        const int64_t nnew = group_scan(it, flag, s.ne);
        for (int64_t e = lid; e < s.ne; e += WG)
          assign_or_relabel(tab, e, s, flag, slot, nbr_all + m * se + s.e0, nodes,
                            row_all + m * se + s.e0, true);
        sycl::group_barrier(grp);
        for (int64_t e = lid; e < s.ne; e += WG)
          assign_or_relabel(tab, e, s, flag, slot, nbr_all + m * se + s.e0, nodes,
                            row_all + m * se + s.e0, false);
        sycl::group_barrier(grp);
        const HopState next{s.lo + s.nf, nnew, s.e0 + s.ne, 0};
        if (lid == 0) {
          st[hop].ne = s.ne;
          st[hop + 1] = next;
        }
        s = next;
      }
    }));
  }

  // Exclusive scan of data[0..n) by one work-group, in chunks; writes data[n] = total and
  // returns it (to every work-item).
  static int64_t group_scan(const sycl::nd_item<1>& it, int64_t* data, int64_t n) {
    const auto grp = it.get_group();
    const int64_t WG = static_cast<int64_t>(it.get_local_range(0)), lid = it.get_local_id(0);
    constexpr int64_t P = 4;
    int64_t carry = 0;
    for (int64_t base = 0; base < n; base += WG * P) {
      const int64_t lo = base + lid * P;
      int64_t v[P], s = 0;
      for (int64_t j = 0; j < P; ++j) {
        v[j] = lo + j < n ? data[lo + j] : 0;
        s += v[j];
      }
      int64_t run = sycl::exclusive_scan_over_group(grp, s, sycl::plus<int64_t>());
      const int64_t total = sycl::group_broadcast(grp, run + s, WG - 1);
      run += carry;
      for (int64_t j = 0; j < P; ++j)
        if (lo + j < n) {
          data[lo + j] = run;
          run += v[j];
        }
      carry += total;
      sycl::group_barrier(grp);
    }
    if (lid == 0) data[n] = carry;
    sycl::group_barrier(grp);
    return carry;
  }

  // Work item g of a hop's count / sampling kernel -> (minibatch m, frontier index i):
  // minibatch-major, or from the locality schedule (build_worklist).
  struct Items {
    const uint32_t* work;   // nullptr: minibatch-major
    const uint32_t* total;  // number of scheduled items (device)
    const HopState* st;
    int64_t H1, NB, K;
    bool get(int64_t g, int64_t& m, int64_t& i) const {
      if (work) {
        if (g >= static_cast<int64_t>(*total)) return false;
        const uint32_t w = work[g];
        m = w >> 24;
        i = w & 0xFFFFFFu;
        return true;
      }
      m = g / NB;
      i = g % NB;
      return m < K && i < st[m * H1].nf;
    }
  };

  // Locality schedule for one hop: counting sort of the K frontiers' (minibatch, index)
  // pairs by node-id bucket (kBuckets buckets of 2^bshift_ consecutive ids). Order within a
  // bucket is whatever the atomics give; it only changes when, not what, each item writes.
  void build_worklist(int K, const HopState* st, int64_t NB) {
    if (K * NB >= (int64_t{1} << 32)) throw std::invalid_argument("schedule: too many work items");
    using Atom = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>;
    uint32_t *cnt = bcnt_, *cur = bcur_, *work = work_;
    const int32_t* nodes = nodes_;
    const int64_t sn = sn_, H1 = hops_ + 1, B = kBuckets;
    const int sh = bshift_;
    rec("schedule", q_.memset(cnt, 0, B * sizeof(uint32_t)));
    rec("schedule", q_.parallel_for(sycl::range<1>(K * NB), [=](sycl::id<1> g) {
      const int64_t m = g / NB, i = g % NB;
      const HopState s = st[m * H1];
      if (i >= s.nf) return;
      Atom(cnt[static_cast<uint32_t>(nodes[m * sn + s.lo + i]) >> sh]).fetch_add(1u);
    }));
    rec("schedule", q_.parallel_for(sycl::nd_range<1>{1024, 1024}, [=](sycl::nd_item<1> it) {
      sycl::joint_exclusive_scan(it.get_group(), cnt, cnt + B, cur, sycl::plus<uint32_t>());
      sycl::group_barrier(it.get_group());
      if (it.get_local_id(0) == 0) cur[B] = cur[B - 1] + cnt[B - 1];
    }));
    rec("schedule", q_.parallel_for(sycl::range<1>(K * NB), [=](sycl::id<1> g) {
      const int64_t m = g / NB, i = g % NB;
      const HopState s = st[m * H1];
      if (i >= s.nf) return;
      const uint32_t pos =
          Atom(cur[static_cast<uint32_t>(nodes[m * sn + s.lo + i]) >> sh]).fetch_add(1u);
      work[pos] = (static_cast<uint32_t>(m) << 24) | static_cast<uint32_t>(i);
    }));
  }

  // Floyd sampling of one frontier node by one sub-group (same draws as Sampler). rid is the
  // node's Philox id (the caller's id on a reordered graph); cold selects the streaming
  // cache hint for its neighbour reads.
  template <class SG>
  static void sample_node(const SG& sg, int32_t v, uint32_t rid, bool cold, int32_t target,
                          int64_t o, int64_t k, uint64_t key, uint32_t hop,
                          const int64_t* indptr, const int32_t* indices, int64_t* epos,
                          int32_t* col, int32_t* nbr) {
    const int64_t lane = sg.get_local_linear_id();
    const int64_t start = indptr[v], d = indptr[v + 1] - start;
    if (d <= k) {
      for (int64_t j = lane; j < d; j += kSG) {
        epos[o + j] = start + j;
        col[o + j] = target;
        nbr[o + j] = load_nbr(indices, start + j, cold);
      }
      return;
    }
    const int64_t jt = d - k + lane;
    int64_t cand = 0;
    if (lane < k)
      cand = bounded(draw_u32(key, hop, rid, static_cast<uint32_t>(lane)),
                     static_cast<uint64_t>(jt + 1));
    int64_t mine = -1;
    for (int64_t t = 0; t < k; ++t) {
      const int64_t c = sycl::select_from_group(sg, cand, t);
      const bool dup = sycl::any_of_group(sg, lane < t && mine == c);
      if (lane == t) mine = dup ? jt : c;
    }
    if (lane < k) {
      epos[o + lane] = start + mine;
      col[o + lane] = target;
      nbr[o + lane] = load_nbr(indices, start + mine, cold);
    }
  }

  // Label-early insert: known nodes are labelled now (slot = -1), new ones record their
  // first edge. View: DirectView or GenHashView (same interface).
  template <class View>
  static void insert_edge(const View& h, int64_t e, int32_t key, int32_t* row,
                          int32_t* slot) {
    const int64_t s = h.insert(key);
    const int32_t v = h.get_val(s);
    if (v >= 0) {
      row[e] = v;
      slot[e] = -1;
      return;
    }
    slot[e] = static_cast<int32_t>(s);
    h.min_first(s, e);
  }

  // assign: winners (rank changes in the scanned flags) get local ids; relabel: repeats of
  // nodes new in this hop read the id their winner wrote.
  template <class View>
  static void assign_or_relabel(const View& h, int64_t e, const HopState& s,
                                const int64_t* flag, const int32_t* slot, const int32_t* nbr,
                                int32_t* nodes, int32_t* row, bool assign) {
    const bool winner = flag[e + 1] != flag[e];
    if (assign) {
      if (!winner) return;
      const int64_t local = s.lo + s.nf + flag[e];
      nodes[local] = nbr[e];
      h.set_val(slot[e], static_cast<int32_t>(local));
      row[e] = static_cast<int32_t>(local);
    } else if (slot[e] >= 0 && !winner) {
      row[e] = h.get_val(slot[e]);
    }
  }

  sycl::queue& q_;
  const int64_t* indptr_;
  const int32_t* indices_;
  int64_t n_;
  std::vector<int> fan_;
  int64_t b_max_;
  int kmax_, k_ = 0, hops_ = 0;
  int64_t mn_ = 0, me_ = 0, sn_ = 0, se_ = 0, hop_e_max_ = 0;
  size_t pstride_ = 0;
  int32_t *nodes_ = nullptr, *col_ = nullptr, *nbr_ = nullptr, *row_ = nullptr, *slot_ = nullptr,
          *bsz_ = nullptr, *seeds_ = nullptr;
  int64_t *take_ = nullptr, *epos_ = nullptr, *flag_ = nullptr, *partial_ = nullptr;
  uint32_t* dtable_ = nullptr;  // direct tables: K * n slots of 8 bytes (D22)
  uint64_t* htable_ = nullptr;  // hash tables: K * hcap_ slots of 16 bytes (D25)
  int64_t hcap_ = 0;
  HopState* state_ = nullptr;
  uint64_t* keys_ = nullptr;
  uint32_t gen_ = 0;
  std::vector<HopState> hs_;
  Reorder ro_;
  int64_t hubs_ = 0;
  bool hints_ = false, sched_ = false;
  uint32_t *work_ = nullptr, *bcnt_ = nullptr, *bcur_ = nullptr;
  int bshift_ = 0;
  // Profiling (set_profile): every command of the last run(), tagged with stage and hop.
  struct Rec {
    const char* name;
    int hop;
    sycl::event e;
  };
  bool prof_ = false;
  int cur_hop_ = -1;
  std::vector<Rec> recs_;
  sycl::event rec(const char* name, sycl::event e) {
    if (prof_) recs_.push_back({name, cur_hop_, e});
    return e;
  }
};

}  // namespace gnnsample
