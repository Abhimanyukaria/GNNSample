# Research notes: GPU neighbor sampling for GNN training

Status as of 2026-10-24. Sources were checked by web research; items marked
**(unverified)** could not be confirmed from a primary source.

## 1. The workload

GraphSAGE minibatch training with fanout **[15,10,5]**, 3 layers, batch of B seed nodes:

- Hop 1 samples ≤ 15 in-neighbors per seed, hop 2 ≤ 10 per hop-1 node, hop 3 ≤ 5 per
  hop-2 node. Sampling is **without replacement**; a node with degree ≤ k keeps all its
  neighbors.
- Between hops the new frontier is **deduplicated** and **relabeled** to local IDs
  (0..n-1), which is what the model's sparse blocks need.
- Upper bound on sampled edges per minibatch: B·(15 + 15·10 + 15·10·5) = **915·B**, so
  ≈ 0.94 M edges for B = 1024. Dedup shrinks the frontiers on small or clustered graphs.
- Features of every unique node in the last frontier are then gathered:
  n_unique × d × 4 bytes. With d = 128 and ~400 K nodes, that is ~200 MB per minibatch,
  versus ~5–8 MB for the sampled topology. This is where the "gather moves ~40× more bytes
  than sampling" figure (BGL) comes from.

Why fanout sampling works: a uniform sample of k neighbors gives an unbiased estimate of
mean aggregation, with cost bounded by k instead of the degree. That bounds the work on
power-law graphs, where some nodes have millions of neighbors.

Why it struggles:
- **Exponential growth.** Work grows as ∏kᵢ, so a 3rd layer costs 5× the first two
  combined.
- **Variance on hubs.** High-degree nodes keep a tiny fraction of their neighbors.
- **Redundancy.** Neighborhoods of different seeds overlap, which forces dedup work and
  repeated reads of the same adjacency lists.
- **Scattered memory access.** Neighbor IDs are effectively random, so gathers miss cache.

## 2. Where training time goes (published measurements)

| Source | Setup | Breakdown |
|---|---|---|
| gSampler (SOSP'23), Table 1 | ogbn-products, GraphSAGE | sampling = 96.2% (PyG-CPU), 70.1% (DGL-CPU), 45.8% (DGL-GPU) |
| SALIENT (MLSys'22) | PyG, V100, [15,10,5], B = 1024 | products: batch prep 46%, transfer 26%, compute 28%; papers100M: 37/35/28 |
| BGL (NSDI'23) | DGL | data I/O + preprocessing 83% of time; GPU utilisation ≤ 15%; ~5 MB subgraph vs ~195 MB features per products minibatch |
| PaGraph (SoCC'20) | GPU feature cache | cuts data loading by up to 96.8%, giving up to 4.8× speedup |

Takeaways:
- With CPU samplers, sampling dominates.
- Even with GPU sampling it is about half of the step time.
- Once sampling is fast, feature gathering and transfer dominate.

We re-measure this on our own hardware (Phase 1) before relying on it.

## 3. Baselines today

| System | Version / status | Where sampling runs |
|---|---|---|
| PyG + pyg-lib | PyG 2.8.1 (Oct 2026), pyg-lib 0.9.0 (Sep 2026) | **CPU only**: pyg-lib's CUDA sampler folder only has random walk |
| DGL | v2.4.0; last commit on master Jul 2025; NVIDIA dropped its DGL containers after 25.08 | GraphBolt (DGL ≥ 2.1): GPU sampling + UVA + GPU feature cache, **CUDA only**; claims up to 5.5× over PyG (products, [15,10,5], DGX-A100). With-replacement sampling is not GPU-accelerated |
| cuGraph-PyG / WholeGraph | active (rapidsai/cugraph-gnn) | CUDA only. cuGraph-DGL was removed in 25.08 |
| Intel XPU | — | no GPU neighbor sampler found **(unverified)**; pyg-lib on XPU only dispatches `index_sort` |

Sources: github.com/pyg-team/pytorch_geometric/releases,
github.com/pyg-team/pyg-lib/tree/master/pyg_lib/csrc/sampler/cuda, github.com/dmlc/dgl,
dgl.ai/release/2024/03/06/release.html, docs.nvidia.com/datascience/notices/rsn0046/,
github.com/rapidsai/cugraph-gnn.

## 4. Prior GPU sampling systems

| System | Key idea | Reported result |
|---|---|---|
| C-SAW (SC'20) | warp-centric parallel selection, collision migration, out-of-memory batching | 10× over KnightKing |
| NextDoor (EuroSys'21) | transit-parallelism: group work by the vertex being expanded, cache its edges in shared memory | 26–50× over KnightKing for random walks; **only 1.2–1.3× end-to-end for GraphSAGE** |
| Skywalker (PACT'21) | GPU alias-method biased sampling, load balancing | beats C-SAW/NextDoor (per gSampler) |
| SALIENT (MLSys'22) | fast CPU sampler + pinned slicing + pipelined transfer | 3–3.4× over PyG |
| GNNLab (EuroSys'22) | separate sampler GPUs and trainer GPUs; pre-sampling-based feature cache | 2.4–9.1× over DGL **(numbers from search snippets)** |
| Quiver | UVA sampling + hot-feature GPU cache for PyG | serving numbers only; repo looks stale |
| DSP (PPoPP'23) | topology + hot features across GPUs over NVLink; collective sampling | up to 26× vs PyG/DGL/Quiver |
| gSampler (SOSP'23) | matrix-centric API, data-flow IR, **kernel fusion within a layer** | 6.54× average (1.14–32.7×) over the best of DGL, PyG, Skywalker, Gunrock, cuGraph |
| GIDS (VLDB'24) | GPU reads graph/features from SSD directly (BaM) | up to 582× over DGL mmap |
| Distributed matrix sampling (MLSys'24) | sampling as SpGEMM, many minibatches at once | 2.5× over Quiver on 128 GPUs |
| FastGL (2024 preprint) | fused-map sampler, match-reorder | 11.8× / 2.2× / 1.5× over PyG / DGL / GNNLab |
| FuseSampleAgg (Nov 2025) | one CUDA kernel for sampling + mean aggregation | up to 51× vs DGL on products 10-10; 0.36× on Reddit 25-10 |

## 5. Experimental conventions we follow

- Fanout [15,10,5], 3 layers, hidden 256, batch 1024 (the de-facto standard: SALIENT,
  DSP, BGL, GraphBolt).
- Sampling without replacement.
- Metric: **sampling time per epoch**, as the mean of the last 5 of 6 epochs (gSampler
  protocol), plus epoch time. We add sampled edges/second as a normalised secondary
  metric; it is not standard **(unverified)**.
- Graphs without features (Friendster, Twitter): random 128-d float features; a random 1%
  of nodes as training seeds (gSampler); random labels. Timing only.

## 6. Datasets

| Graph | Nodes | Edges | Download | Source |
|---|---|---|---|---|
| ogbn-products | 2,449,029 | 61.86 M undirected | 1.48 GB zip | ogb.stanford.edu |
| soc-LiveJournal1 | 4,847,571 | 68,993,773 directed | 260 MB gz | snap.stanford.edu |
| com-Orkut | 3,072,441 | 117,185,083 undirected | 447 MB gz | snap.stanford.edu |
| twitter-2010 | 41,652,230 | 1,468,365,182 arcs | 2.5 GB BVGraph | law.di.unimi.it (needs WebGraph to decode) |
| com-Friendster | 65,608,366 | 1,806,067,135 undirected | 9.37 GB gz | snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz |

In-memory CSC (int64 indptr + int32 indices):
- Friendster, symmetrised: 3.61 B × 4 B + 65.6 M × 8 B ≈ **15.0 GB**. That does not fit
  in this laptop's 15.5 GB of shared RAM.
- Twitter (in-edges): 1.47 B × 4 B + 0.33 GB ≈ **6.2 GB**, which is borderline.
