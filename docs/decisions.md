# Decision records

Each record states the choice, why we made it, the strongest alternative, what happens if
we are wrong, and the measurement that settles it. Status is one of:
- **open**: initial position, not yet measured
- **confirmed**: measured; the choice held
- **reversed**: measured; we switched
- **fixed by user**: not up for re-evaluation

---

## D1. Fanout [15,10,5], 3 layers, without replacement — *fixed by user*
- **Why:** this is the de-facto standard (SALIENT, DSP, BGL, GraphBolt), so our numbers
  are directly comparable with published ones. Without replacement is the PyG/DGL default.
- **Alternative:** sweep fanouts ([25,10], [10,10,10]). This was rejected to keep the
  study focused.
- **If wrong:** our optimisations could be tuned to one fanout shape. Mitigation: the
  kernels take fanouts at run time, not as constants.

## D2. SYCL (Intel DPC++) instead of CUDA — *open*
- **Why:**
  - It lets us develop and test locally on the Intel iGPU.
  - It is one codebase for Intel, NVIDIA (CUDA backend) and AMD.
  - No GPU neighbour sampler exists for XPU.
- **Alternative:** CUDA, which has the best tooling (Nsight) and the same platform as
  the state-of-the-art baselines.
- **If wrong:** SYCL on NVIDIA could be slower than equivalent CUDA, which would weaken a
  head-to-head against GraphBolt or gSampler.
- **Settled by:** Phase 4b. Write the hottest kernel in plain CUDA and compare it with the
  SYCL version on the same NVIDIA GPU.

## D3. CSC layout: int64 `indptr`, int32 `indices` — *open*
- **Why:**
  - Sampling reads in-neighbours, and CSC stores those contiguously.
  - int32 neighbour IDs halve the bytes per sampled edge. Every target graph has fewer
    than 2³¹ nodes.
  - int64 offsets are required because Friendster has more than 2³¹ edges.
- **Alternative:** int64 IDs everywhere, which is PyG's default. That costs 2× memory
  and bandwidth.
- **If wrong:** only if a graph has 2³¹ or more nodes, which none of ours do.

## D4. Without-replacement algorithm: Floyd's for small k, partial Fisher–Yates otherwise — *open*
- **Why:**
  - Floyd's algorithm draws k distinct indices out of d in O(k) RNG calls, independent of
    the degree d. With k ≤ 15 the membership check runs over at most 15 registers.
  - This keeps hubs (degree up to ~10⁶) from becoming stragglers.
- **Alternatives:**
  - Reservoir sampling: O(d) per node, so it is terrible on power-law graphs.
  - Rejection with a bitmap: needs O(d) scratch.
  - Sampling with replacement: trivial, but changes the semantics.
- **If wrong:** the k² membership checks could cost more than expected. Measure Floyd
  against reservoir and Fisher–Yates on the hub-heavy bins.

## D5. Work mapping: degree-binned hybrid — *open*
- **Why:**
  - Nodes with degree ≤ k just copy their neighbour list (a coalesced copy).
  - Nodes with degree > k run Floyd's algorithm.
  - Binning separates these two code paths, so they don't diverge inside one sub-group.
- **Alternative:** one work-item per seed. It is the simplest option, and its divergence
  might be cheap at k ≤ 15.
- **If wrong:** the cost of binning (an extra scan or partition) could exceed what it
  saves. Ablate it.
- **Note:** sub-group size is a compile-time parameter: 16 on Intel Xe, 32 on NVIDIA.

## D6. Frontier dedup and relabel: GPU open-addressing hash table — *open*
- **Why:**
  - Inserting a node returns its local ID in one pass, so dedup and relabel happen
    together.
  - Its cost scales with the frontier size, not with log(frontier) passes as a sort
    would.
  - DGL uses the same approach.
- **Alternative:** sort + unique + searchsorted. It is deterministic and simpler, but
  needs several passes over global memory.
- **If wrong:** atomic contention on the hash table could make it slower than sorting.
  Implement both and measure.
- **Linked angle:** keeping one table alive across all 3 hops is the "fused pipeline"
  idea (main differentiation candidate 1).

## D7. RNG: counter-based Philox-4x32-10 keyed on (seed, hop, node, draw) — *open*
- **Why:** it is stateless, so every work-item can generate its own random numbers
  independently. It gives the same samples on any backend and with any work mapping, so
  correctness tests compare our output bit-for-bit with a reference sampler.
- **Alternative:** oneMKL device RNG or per-thread state (curand-style). Those are faster
  to set up, but the results depend on the work mapping.
- **If wrong:** if Philox costs a noticeable share of kernel time, switch to a cheaper
  counter-based hash such as squares64. Measure it.

## D8. Baseline protocol — *open*
- **Why:**
  - Exclude warm-up and JIT: the first epoch is discarded and the mean of epochs 2–6 is
    reported (the gSampler protocol).
  - Each configuration is repeated 5 times; we report the median and IQR.
  - Hardware, versions and thread counts are recorded in every results JSON.
  - CPU baselines get a thread-count sweep so their best setting is the one compared.
- **Alternative:** a single run, or a fixed thread count. That is easier, but it invites
  an unfair "we beat a mistuned baseline" claim.

## D9. Bottleneck gate before kernel work — *open*
- **Why:** the literature says sampling is 46–96% of step time, but feature gathering can
  take over once sampling is fast. Measure the split for PyG on ogbn-products on our own
  hardware first.
- **Gate:** if sample + dedup < ~20% of step time, refocus on fused sample + gather.

## D10. Multi-hop semantics: PyG-style (expand only newly discovered nodes) — *open*
- **Why:** our training integration and accuracy check use PyG. The output must be a drop-in
  replacement for PyG's `NeighborLoader`, which expands only the nodes first discovered at
  the previous hop.
- **Alternative:** DGL-style blocks. Every layer re-samples all of its destination nodes,
  including nodes already expanded, so it does more work per minibatch.
- **If wrong:** comparisons with DGL would be unequal. Mitigation: report sampled edges per
  minibatch next to time, and compare edges/s as well as time.
- **Reproducibility detail:** the multiply-shift range reduction has a bias of at most
  d/2³² (< 0.1% for degrees up to 4M). This is accepted; `tests/test_reference.py` runs a
  chi-square uniformity check.

### D9 measurement 1 — 2026-10-24, laptop iGPU (results/profile_pyg_products_xpu.json)
PyG 2.8.1 + pyg-lib 0.9.0 on ogbn-products, GraphSAGE 3×256, [15,10,5], batch 1024,
model on the Intel iGPU (0x7d41), 7 CPU threads, serial loop, median over 100 batches
(10 warm-up batches excluded). Each minibatch averages 409 K nodes, 715 K edges and
164 MB of features.

| stage | median | share |
|---|---|---|
| sample (pyg-lib, CPU) | 0.166 s | 7.4% |
| feature gather (CPU) | 0.308 s | 13.8% |
| host→device copy | 0.066 s | 3.0% |
| train step (iGPU) | 1.702 s | 75.9% |

**Gate result on this hardware: sampling is 7% of the step, below the 20% threshold.**
The model step dominates. It is roughly 10–20× slower than the FLOP count would suggest
(PyG runs every layer over all 409 K nodes, and XPU scatter/aggregate kernels are slow).
In a real loader with workers, sampling would hide entirely behind training.

This contradicts the data-centre GPU literature (SALIENT: compute 28%; gSampler: sampling
46–96%) because the iGPU's compute is so much slower. So on this laptop an end-to-end win
from a faster sampler cannot be shown. The sampler claim has to rest on sampling
throughput here, plus end-to-end runs on an NVIDIA GPU later. Gather is 2× the cost of
sampling, which supports the fused sample + gather angle.

### D9 measurement 2 — why the train step took 1.7 s, and the corrected split
`bench/train_step.py` (20 fixed products minibatches, train step only; results/train_step_products.json):

| variant | device | step | fwd | bwd |
|---|---|---|---|---|
| scatter, no trimming (as in measurement 1) | iGPU | 1.91 s | 0.72 | 1.17 |
| scatter, **layer trimming** | iGPU | **0.30 s** | 0.14 | 0.15 |
| sparse CSR matmul (± trimming) | iGPU | fails: XPU sparse matmul densifies the matrix (asks for 198–642 GB) | | |
| scatter, trimming | CPU (14 thr) | 1.83 s | 0.85 | 0.78 |
| sparse CSR matmul, trimming | CPU | 1.73 s | 0.46 | 1.09 |
| scatter, no trimming | CPU | OOM-killed (7.5 GB container) | | |

**Cause:** without trimming, all 3 layers run over all ~410 K sampled nodes, but 70% of
them (~287 K 3rd-hop leaves) are only inputs. Their layer-2/3 outputs are never used.
Kernel profile of the untrimmed step: dense `gemm`/`mm` was 63% of GPU time;
gather/scatter (`index_add`)/div over the same oversized tensors was ~30%.
Trimming (`gnnsample/models.py`, same idea as PyG's `trim_to_layer`) gives a **6.3×**
faster step. The iGPU is ~6× faster than the CPU for this model.

**Superseded by measurement 3:** the gather number below came from a bug in our data
preparation (column-major feature table). It is kept for the record.

End-to-end rerun with trimming, PyG-style `index_select` gather, and edge orientation
checked (median of 100 batches):

| stage | median | share |
|---|---|---|
| sample (pyg-lib, CPU) | 0.137 s | 17% |
| feature gather (CPU `index_select`, 164 MB) | 0.367 s | **47%** |
| host→device copy | 0.050 s | 6% |
| train step (iGPU, trimmed) | 0.232 s | 30% |

**Gate, re-evaluated: data preparation (sample + gather + copy) is 70% of the step on
this laptop too, once the model is configured properly.** This matches the literature.
Measurement 1 was an artifact of untrimmed layers.

Gather is now the single largest stage. It runs at only ~0.45 GB/s (1.1 M random 400 B
rows/s), far below memory bandwidth, so it gets its own micro-benchmark before we decide
how to fuse it with sampling.

### D9 measurement 3 — the slow gather was our bug; corrected split
Micro-benchmarks: `bench/gather.py`, `bench/gather_access.py`, `bench/gather_bisect.py`.
The same `index_select` of ~410 K rows took 20 ms in one script and 290–480 ms in another.
Cause: `data/prepare.py` saved `feat.npy` from `pandas.to_numpy()`, which is **column-major**
(torch strides (1, 2449029)). A row gather then touches 100 cache lines ~10 MB apart per
row. Real PyG (OGB loader) has a row-major table, so this was our artifact, not PyG's cost.
Fixed in `prepare.py` (`np.ascontiguousarray`). `graph.load` now refuses non-row-major
extras.

Ruled out along the way (measured): allocating a fresh output (33 ms to allocate and touch
164 MB), swapping (pswpin/pswpout unchanged during the run), repeated vs fresh indices
(20.3 vs 20.5 ms), and random vs sampled vs sorted indices (22.5 / 20.5 / 20.7 ms).

Corrected end-to-end profile (row-major table, trimmed model, iGPU, median of 100 batches;
results/profile_pyg_products_xpu_trim.json):

| stage | median | share |
|---|---|---|
| sample (pyg-lib, CPU, serial) | 0.142 s | **30%** |
| feature gather (CPU, 7 threads, 164 MB) | 0.038 s | 8% |
| host→device copy (pageable) | 0.052 s | 11% |
| train step (iGPU, trimmed) | 0.238 s | 51% |

**Gate: data preparation is 49% of the step; sampling alone is 30%, above the 20%
threshold, so the sampler focus stands.** Gather + copy together are 19%.

Gather, ~410 K rows × 400 B (results/gather_products.json; GB/s = bytes read + written):

| variant | time | GB/s |
|---|---|---|
| CPU index_select, 1 / 4 / 7 / 14 threads | 47.9 / 24.0 / 23.4 / 30.5 ms | 6.8 / 13.7 / 14.0 / 10.7 |
| CPU gather + pageable copy to iGPU | 75.1 ms | 4.4 |
| pageable / pinned copy of 164 MB to iGPU | 34.7 / 18.2 ms | |
| torch on iGPU, table in device memory | 21.8 ms | 15.1 |
| SYCL on iGPU, table in device / **host** / shared USM | 24.2 / **21.9** / 24.3 ms | 13.6 / **15.0** / 13.5 |
| SYCL contiguous rows (ceiling), device / host | 12.7 / 13.1 ms | 25.8 / 25.1 |

Readings:
- On the iGPU, reading the table from **host USM (zero-copy) costs the same as device
  memory**. The table need not be duplicated, which matters with ~8 GB for everything.
- A GPU gather leaves the rows already on the device. That replaces CPU gather + copy
  (~75–90 ms) with ~22 ms and frees the CPU threads.
- Sampled rows reach ~60% of the contiguous ceiling. One minibatch touches 79% of the
  table's 4 KB pages, so there is little locality left to exploit.
- More than 7 CPU threads is slower (E-core/LP-E-core contention).

## D11. Fusing gather with the sampler: pipeline fusion over zero-copy host USM — *open*
- **Choice:**
  - The feature table stays in host USM (zero-copy on the iGPU).
  - The sampler's final node list stays in device USM.
  - The gather kernel runs on the same in-order queue right after the last hop, with no
    host round-trip.
  - The output goes to PyTorch as a device tensor (DLPack).
- **Why:**
  - The measurements above: no CPU gather, no copy, no duplicate table.
  - Expected saving per step is ~(38 + 52) − 22 ≈ 68 ms of 469 ms on this laptop. To be
    measured.
- **Alternative 1, kernel fusion:** write rows inside the last-hop sampling kernel. Rejected
  for now:
  - Gather needs the final deduplicated node list, which only exists after the last
    dedup.
  - The node list is ~1.6 MB, so the extra read that fusion would save is <1% of the
    164 MB moved.
  - Revisit only if launch/sync overhead shows up in profiles.
- **Alternative 2, CPU gather into pinned memory + async copy:** ~23 + 18 ms. It's cheaper
  than today's path but still slower than a GPU gather, and it occupies CPU threads.
- **If wrong:**
  - On a discrete NVIDIA GPU, host USM means reading over PCIe (DGL's "UVA" mode,
    ~10–25 GB/s vs ~1–3 TB/s device memory).
  - There the choice becomes device-resident features or a hot-node cache. This decision
    is therefore per-device, settled in Phase 4b.

## D12. Measurement method for optimizations: paired, interleaved A/B — *confirmed*
- **Why:** the iGPU's speed drifts between runs. The same v1 benchmark gave 30.8, 33.0,
  48.0 and 65.1 ms per batch on products, while pyg-lib on the CPU stayed within
  129–134 ms. Separate before/after runs cannot resolve 5–20% effects.
- **Method:**
  - Every optimization is a run-time option bit in one binary.
  - `bench/ablate.py` samples every minibatch with each variant in rotating order, in the
    same process.
  - The effect is the median over minibatches of t_variant / t_baseline.
  - Correctness is checked by `tests/test_gpu_sampler.py`: every option combination is
    bit-identical to the NumPy reference (52 tests).
- **Stage timing:** use GPU timestamps of barrier markers (`set_profile(2)`), not host
  waits. Host waits inflated small stages and made the scans look like a hotspot.
- **Keep rule:** keep an optimization if it is consistently below ~0.95 on the graphs.
  Drop it otherwise, and record why.

## D13. Phase 3 sampler optimizations — results (iGPU, batch 1024, [15,10,5])
Paired ratios vs the previous best (`results/ablation.jsonl`):

| option | products | LiveJournal | Orkut | verdict | why |
|---|---|---|---|---|---|
| packed hash slots (16 B: key, local id, first edge) | 0.83 | 0.93 | 0.85 | **keep** | one cache line per edge instead of three |
| fast scan (single-kernel small / coalesced large) | 0.99 | 1.01 | 1.00 | drop | scans were never the bottleneck; host-wait profiling had misled us |
| clear only used slots | 1.20 | 1.14 | 1.22 | drop | 410 K scattered writes cost more than a 32 MB streaming fill |
| **no-sync** (upper-bound launches, counts on device, 1 sync per batch) | 0.75 | 0.82 | 0.85 | **keep** | an 8-byte readback + wait costs ~0.6 ms under WSL2 (`launch_overhead`); v1 did 6 per batch |
| Philox reuse (4 words per call) | 1.02 | 1.01 | 1.01 | drop | sampling is memory-latency bound, not RNG bound |
| assign from scan (flag[e+1] != flag[e]) | 0.98 | 0.99 | 0.98 | keep | small but consistent; free |
| 32-bit slot indices | 0.99 | 0.99 | 0.99 | keep | small but consistent; free |
| **generation tags, 16 B slots** (no per-batch reset) | 0.90 | 0.82 | 0.95 | **keep** | removes the 32 MB reset; a first 32 B-slot version was neutral on products/Orkut (2× cache lines), so slots were packed to 16 B with 8-bit generations |

Default = packed | no-sync | assign-from-scan | slot32 | gen-tags (`loader.DEFAULT_OPTIONS`).
- **vs our v1** (paired): **1.60× products, 1.89× LiveJournal, 1.58× Orkut**.
- **vs single-process pyg-lib, per batch:** **8.9×, 5.5×, 7.5×** (17.1 / 8.7 / 22.6 ms).

Bugs found and fixed on the way:
- An empty seed batch left the no-sync device state uninitialised (GPU fault on Orkut,
  whose 1% seeds give only 30 full batches).
- The iGPU's 1.07 GB single-allocation cap broke large feature tables. Tables are now
  chunked at 512 MB.

## D14. End-to-end training step, ours vs PyG at its best (products, iGPU) — *measured*
`bench/train_e2e.py`: GraphSAGE 3×256, [15,10,5], batch 1024, trimmed layers, 60
minibatches, one configuration per process. PyG uses DataLoader workers, each running
pyg-lib + `index_select` into shared memory (what NeighborLoader does), with pinned,
non-blocking copies. Ours is the SYCL sampler + GPU gather from host USM, serial.

| config | median ms/step | mean ms/step | CPU cores used |
|---|---|---|---|
| PyG, 0 workers | 612.6 | 635.0 | ~1 |
| PyG, 2 workers | 349.3 | 423.2 | ~3 |
| PyG, 4 workers | 309.2 | 613.4 | ~5 |
| PyG, 6 workers | 334.1 | 689.8 | ~7 |
| **ours** | **257.4** | **259.3** | ~1 |

Epoch time follows the mean:
- **Ours is 1.63× faster than PyG's best (2 workers).**
- **On the median it is 1.20× faster than PyG's best (4 workers).**
- **It uses ~1 CPU core instead of 3–7.**

Caveats:
- PyG with ≥4 workers has heavy stalls (mean ≈ 2× median), most likely memory pressure
  in the 7.5 GB container. Each worker holds copies, plus 164 MB of shared memory per
  in-flight batch. With more RAM, PyG's mean would likely move toward its median.
- On this iGPU our sampler shares the chip with training, so its time adds to the step.
  PyG workers instead overlap sampling with training. Our margin here is smaller than the
  data-preparation speedup (~43 vs ~232 ms) for exactly that reason.
- On a discrete GPU with fast training, the literature says data preparation dominates.
  There our margin should grow; that is Phase 4b's test.
- PyG with 0 workers pins memory in the main process, which costs more than the earlier
  serial profile (448 ms).

## D15. Accuracy parity, ours vs PyG — *confirmed*
`bench/accuracy.py` + `bench/accuracy_report.py` (results/accuracy/*.json):
- **Setup:** ogbn-products, GraphSAGE 3×256, [15,10,5], batch 1024, Adam lr 1e-3,
  5 epochs, 3 seeds per sampler.
- **Controlled per seed:** model init, seed order and hyperparameters are identical; only
  the sampler differs.
- **Scoring:** the same evaluator (pyg-lib sampled inference) for every model, on the full
  val set and a fixed random 50 K-node test subset.
- **PyG side:** 2 DataLoader workers, overlapped across epochs.

| sampler | val acc (last epoch) | test acc (50 K) | train s/epoch (mean of epochs 2–5) |
|---|---|---|---|
| ours | 0.9013 ± 0.0036 | 0.7679 ± 0.0082 | 55.3 |
| PyG | 0.8994 ± 0.0052 | 0.7669 ± 0.0096 | 72.5 |

- Per-seed test: 0.7770 / 0.7766, 0.7657 / 0.7667, 0.7610 / 0.7574 (ours / PyG).
- Welch t-test: val +0.0019 (p = 0.63), test +0.0010 (p = 0.90).
- Mean val per epoch tracks within 0.002 throughout (0.8734 / 0.8734 at epoch 1).

**Verdict:** no accuracy difference within seed noise. Training is 1.31× faster per epoch.
Test accuracy (~0.77) is in line with published GraphSAGE neighbour-sampling results on
products (~0.78 after more epochs).

Where the 1.31× comes from (steady-state step ~285 vs ~378 ms):
- Model compute on the iGPU is ~235 ms of both. Our data preparation is ~50 ms exposed
  vs ~143 ms for PyG.
- Amdahl ceiling on this hardware: ~378 / 235 ≈ 1.6×.
- Part of the gain is the GPU gather (D11), not only the sampler.

### D13 addendum — sub-group cooperative sampling (option 256)
One 16-lane sub-group per frontier node:
- lane t draws Floyd candidate t in parallel;
- a k-step broadcast/any-of loop applies Floyd's sequential collision rule exactly
  (bit-identical, 54 tests);
- all k neighbour loads are then in flight at once; the d ≤ k copy is coalesced.

Paired ratio vs the previous best (option 233), `results/logs/ablate-sg.log`:

| graph | ratio |
|---|---|
| Orkut (avg degree 76) | **0.896** |
| products | 0.959 |
| LiveJournal (avg degree 14) | 0.982 |

The gain grows with degree, as expected for a latency-bound kernel that now issues k loads
in parallel. **Kept**; now part of `DEFAULT_OPTIONS`. The sub-group width is the constant
`kSubGroup` (16 on Intel Xe; set to 32 for NVIDIA).

**Measurement caveat:** this run happened on battery (27%, discharging, Balanced plan).
pyg-lib took 343 ms per products batch instead of ~145 ms. Paired ratios are unaffected
(interleaved), but this run's absolute and vs-pyg numbers are not comparable with earlier
runs and are not quoted. `tools/run_logged.sh` now records the power state in every log.

## D16. Overlap: sample minibatch i+1 while training minibatch i — *kept (small gain on iGPU)*
`gnnsample.loader.PrefetchingGpuSampler`:
- A background thread runs our sampler and gather into one of two preallocated XPU slots
  while PyTorch trains on the other.
- An XPU event recorded after the consumer's step guards slot reuse.
- Bit-identical to serial, including under heavy concurrent PyTorch work on the slots
  (`tests/test_pipeline.py::test_prefetching_matches_serial`).

Interleaved A/B (`bench/overlap_ab.py`, 3 rounds × 20 steps per variant, AC power):

| graph | serial ms/step | overlapped | ratio |
|---|---|---|---|
| products | 316.5 | 303.0 | 0.957 |
| Orkut | 299.8 | 289.6 | 0.966 |
| LiveJournal | 112.6 | 108.4 | 0.963 |

**Why only ~4%:** on the iGPU our sampling kernels and PyTorch's training kernels share the
same 64 EUs and the same memory bandwidth, so only ~10–13 ms of our ~40 ms data path
hides. **If wrong / on NVIDIA:** separate streams on a discrete GPU should hide most of
the data path; to measure in Phase 4b.

## Headline numbers, AC power, final defaults (2026-10-24)
Options: packed | no-sync | assign-from-scan | slot32 | gen-tags | sub-group.

Per-batch sampling, paired on the same 40 batches (`results/logs/headline-sampler.log`):

| graph | ours | our v1 | vs v1 | pyg-lib (1 process) | **vs pyg-lib** |
|---|---|---|---|---|---|
| products | 18.3 ms | 32.4 ms | 1.77× | 148.5 ms | **7.9×** |
| LiveJournal | 9.4 ms | 19.7 ms | 2.02× | 69.5 ms | **7.4×** |
| Orkut | 26.5 ms | 45.2 ms | 1.69× | 168.8 ms | **6.4×** |

GPU stage times on products (ms):
- reset 0.0, seeds 1.9, count + scans 1.8, **sample 3.1**;
- hash-insert 4.2, flag 2.4, assign 3.7, relabel 2.5 — **the hash phases (~13 ms) are now
  the largest cost**.

End-to-end training step on products, 60 minibatches, one config per process
(`results/logs/headline-e2e-*.log`):

| config | median ms/step | mean ms/step |
|---|---|---|
| PyG, 0 workers | 477.3 | 496.1 |
| PyG, 2 workers | 349.3 | 396.6 |
| PyG, 4 workers | 402.5 | 668.2 |
| ours (serial) | 292.7 | 297.3 |
| **ours (overlapped)** | **291.6** | **293.8** |

**vs PyG's best (2 workers): 1.35× on mean step time (sets epoch time), 1.20× on median.**
Ours uses ~1 CPU core and is far steadier (mean ≈ median).

The e2e script synchronises after every step for timing, which limits the overlap here
(≈1% vs ≈4% in `overlap_ab.py`).

## D17. Hash-table phases (insert / flag / assign / relabel) — measured first, then optimized
Edge statistics (`bench/hash_stats.py`, 10 batches; share of all sampled edges):

| graph | neighbour already labelled ("old") | first occurrence ("winner") | repeat of a new node | same key twice in a 16-lane group | table load |
|---|---|---|---|---|---|
| products | 23.8% | 60.5% | 15.7% | 1.6% | 0.16 |
| LiveJournal | 19.1% | 71.0% | 9.9% | 2.9% | 0.09 |
| Orkut | 12.9% | 74.3% | 12.8% | 0.4% | 0.26 |

Rejected before implementation:
- **Warp-aggregated inserts.** At most 0.4–2.9% of inserts share a key within a sub-group,
  so the ceiling is under ~1% of hash time.
- **Fusing relabel into the next hop's kernel.** It saves one ~10–30 µs launch per hop.
  Relabel's ~2.5 ms is random table reads, which fusion does not remove.

Implemented (option bits 512 and 1024), paired A/B vs the previous default (AC):

| option | products | Orkut | LiveJournal | why |
|---|---|---|---|---|
| **label early**: old edges labelled at insert, winners at assign, relabel only repeats | 0.937 | 0.935 | 0.990 | relabel's random reads drop from 100% of edges to the 10–16% repeats |
| **small table**: cap/2 slots (≥ worst-case nodes, never fills) | 0.950 | 0.966 | 1.022 | half the cache footprint at load ≤ 0.5 |
| **both** | **0.927** | **0.920** | **0.907** | kept; now in `DEFAULT_OPTIONS` |

GPU stage times on products (option 489 → 2025):
- **relabel 2.4 → 0.9 ms**, flag 2.3 → 1.8 ms;
- insert 4.3 and assign 3.7 ms are unchanged, as predicted. They are the next hash targets.

All 60 tests pass, including option switches between batches across table sizes.

**Measurement caveat:** pyg-lib's per-batch time swung 114–510 ms between runs on AC
(host CPU contention). Vs-pyg ratios from this run are not quoted; paired ratios are
unaffected.

## D18. Insert and assign: direct-mapped dedup + no node_slot writes — *kept (small)*
Measured first (`bench/hash_layout_stats.py`, inserts replayed on the host with the GPU's
hash):

| graph | probes/insert | inserts that CAS a new key | distinct 64 B lines: hashed vs direct |
|---|---|---|---|
| products | 1.19 | 60.6% | 194 K vs 201 K (1.03×) |
| LiveJournal | 1.09 | 71.1% | 140 K vs 130 K (0.93×) |
| Orkut | 1.49 | 74.3% | 237 K vs 284 K (1.20×) |

Readings:
- No locality gain from indexing by node id: sampled ids are scattered.
- Probing is cheap.
- What direct mapping removes is the 64-bit claim CAS, ~45% of the atomics per insert.
- Separately, assign wrote `node_slot` (8 B per winner), which only the rejected
  clear-used option reads.

Implemented:
- **`kDirectMap`:** node id = slot, 8 B per graph node, generation-tagged, no key, no
  probing, no CAS.
- **`kNoNodeSlot`:** skip the dead write. The table is marked invalid for clear-used
  whenever node_slot was skipped.

65 tests pass. Paired A/B vs the previous default (2025):

| run | products | Orkut | LiveJournal |
|---|---|---|---|
| A/B (each alone: 0.96–1.01, i.e. noise) | 0.950 | 0.977 | 0.916 |
| confirmation 1 (60 batches) | 0.947 | 0.972 | 0.954 |
| confirmation 2 (60 batches) | 0.932 | 0.951 | 1.028 |

- products: −5 to −7% consistently. Orkut: −2 to −5%.
- LiveJournal is inconclusive: ±5% run noise at ~10 ms per batch.
- Each change alone saves ~0.5 ms, below single-run noise; together ~1.2 ms on products.
- GPU stage times on products: insert 4.1 → 3.3 ms, assign 3.6 → 3.0 ms.
- **Kept** (now in `DEFAULT_OPTIONS`).
- **Cost:** device memory grows with the graph (8 B/node: 20 MB products, 39 MB
  LiveJournal, ~0.5 GB Friendster). It falls back to the hash table when generation tags
  cannot be used (> 2^24 nodes or edges per batch).

Remaining per-batch GPU time on products (ms):
- sample 4.0, insert 3.3, assign 3.0, flag 1.6, seeds 1.0, scans 1.1, relabel 0.8, count 0.5.
- The phases are now fairly balanced; further gains need structural changes (e.g. fusing
  flag into insert's second pass), each worth only a few percent on this iGPU.

## D19. Scale to Twitter-2010 on the laptop — *measured*
**Data.** Twitter-2010, LAW's transposed BVGraph (2,434,414,406 bytes).
- Decoded by `docker/webgraph/ToCsc.java` (WebGraph 3.6.12, separate converter image)
  straight into the CSC: the transposed graph's successor lists are in-neighbour rows in
  node order, so nothing is sorted. The conversion took ~4 min.
- Result: 41,652,230 nodes and 1,468,364,884 edges. That is the published 1,468,365,182
  minus 298 self loops, per the prepare.py convention.
- `data/finish_twitter.py` validated span, id range, sorted rows and no self loops.
- Max in-degree 770,155; 5.96 M nodes have zero in-degree.

**Memory lesson (a hang, then the fix).**
- With WSL raised to 10 GB, our first run hung the machine. The Docker API returned
  500s, the VM's resident set fell to 1.8 GB, and Windows had 2 GB free.
- Cause: on WSL2 the iGPU's device memory is allocated on the Windows side, outside the
  VM limit. 5.9 GB of device memory, plus the VM's page cache from reading the 5.9 GB
  file, plus Windows exceeded the 15.5 GB of RAM, so Windows paged the VM out.
- Fix: `_C.Sampler.from_files` streams the CSC to the device in 64 MB chunks and drops
  each chunk from the page cache (`posix_fadvise DONTNEED`). The staging buffer is plain
  heap: `pread` into pinned USM host memory fails on Docker Desktop bind mounts
  (container-local files worked), covered by `test_from_files_on_bind_mount`.
- With the fix, Windows free memory bottomed at 337 MB during the load and then held at
  1.3–2.3 GB. This is the practical ceiling of this laptop.
- `tools/watch_runs.sh` now kills gnnsample containers if Windows free memory drops below
  1 GB, and exits by itself once nothing runs. Stopped Monitors had left 12 orphaned loops.

**Results** (`results/scale.jsonl`, `results/logs/scale-twitter-{pyg,ours3}.log`):
- AC power; batch 1024; [15,10,5]; a random 1% of nodes as seeds (gSampler); 40 batches.
- The two samplers ran in separate processes because of memory, on the same batches.

| | median ms/batch | edges/batch | M edges/s | graph load |
|---|---|---|---|---|
| pyg-lib (CPU, int32 — it accepts int32 since 1.47 B < 2³¹) | 46.4 | 175,392 | 3.8 | 116 s |
| ours v1 | 13.05 | 175,541 | 13.4 | 63 s |
| **ours final (8169)** | **7.93** | 175,541 | **22.1** | 63 s |

- **vs pyg-lib: 5.9×.** Unpaired across processes, both on AC.
- **vs v1: 0.569, paired** (1.76×), in line with the smaller graphs (1.69–2.02×).
- GPU stages (ms): sample 1.6, insert 1.4, assign 1.3, seeds 0.7, flag 0.7, scans 0.5,
  count 0.3, relabel 0.2.
- Batches are lighter than products (175 K vs 556 K edges) because random Twitter seeds
  often have few or no in-neighbours.

**Not done at this scale:**
- End-to-end training: random features for 41.7 M nodes, even 16-d (2.7 GB of
  Windows-side memory), plus the 6.2 GB graph would not fit next to the VM and Windows.
- Friendster (~15 GB of CSC) cannot fit on this laptop at all.

## D20. CPU execution: per-batch splitting does not scale past ~6 threads — *open (proposal)*
Measured (`bench/cpu_scaling.py` via `tools/cpu_sweep.sh`; `results/cpu_scaling.jsonl`):
- Sampling-only throughput at equal thread counts, 2 rounds, battery power, medians.
- **Ours:** the SYCL sampler on the CPU device, one minibatch at a time, its runtime limited
  to T threads (`CL_CONFIG_CPU_TBB_NUM_WORKERS` / `DPCPP_CPU_NUM_CUS`; both work).
- **pyg-lib:** T DataLoader worker processes, one thread each.

| T | products ours / pyg (batches/s) | ratio | Orkut ours / pyg | ratio |
|---|---|---|---|---|
| 1 | 7.4 / 6.7 | 1.09 | 7.8 / 5.2 | 1.50 |
| 2 | 19.3 / 12.7 | 1.52 | 15.9 / 7.9 | 2.00 |
| 4 | 26.1 / 19.3 | 1.35 | 21.1 / 14.6 | 1.45 |
| 6 | **32.4** / 27.1 | 1.20 | **24.6** / 23.3 | 1.06 |
| 8 | 30.9 / 28.1 | 1.10 | 25.5 / 32.9 | 0.77 |
| 14 (all) | 20.6 / **45.9** | **0.45** | 12.7 / **34.6** | **0.37** |

By default the CPU runtime uses all 14 hardware threads, which is the worst setting here.

**Why:**
- Our kernels split one minibatch across all threads, with a barrier at every kernel end
  (~30 per minibatch).
- Hop 1 has only 1024 seeds.
- The Core Ultra 7 265U mixes 2 P-, 8 E- and 2 LP-E cores, so the slowest thread gates
  every barrier.
- pyg-lib parallelises over whole minibatches (independent processes, no
  synchronisation), which scales almost linearly.

**Architecture reading:**
- Per-batch data parallelism is the right design for GPUs: one minibatch must fill the
  device, and latency matters because training shares it.
- On CPUs, batch-level parallelism wins at high core counts.

**Proposal (not started):** a CPU mode with several sampler instances, each with ~4–6
threads on its own queue, sharing one read-only graph copy. This would be pyg-lib's
scaling with our per-core advantage (1.1–2.0× at 1–6 threads). It would be A/B-tested
against pyg-lib workers with this same sweep.

**Interim:** for CPU use, cap the runtime at about 6 threads (`DPCPP_CPU_NUM_CUS=6`).

## D21. Latency anatomy of one minibatch (iGPU), sized by order of magnitude — *measured*
Tools:
- `bench/latency_anatomy.py`: profile mode 3. Every no-sync kernel's event goes through
  `Sampler::track()`, giving GPU start/end per stage. The queue's
  `ext_oneapi_get_last_event()` returned meaningless timestamps on this runtime and is not
  used.
- `cpp/tools/launch_overhead.cpp`: pointer chase and random-read microbenchmarks.

All runs were on battery (26–31%), so compare shares, not absolute times.

| component (products, batch 1024) | time | share | order of magnitude |
|---|---|---|---|
| GPU kernel execution | 11.6 ms | ~75% | 10 ms |
| of which hop 3 (sample 2.7, insert 1.8, assign 1.7, flag 0.9, scan 0.4, relabel ~0.4, count 0.3) | ~8 ms | ~55% | 10 ms |
| of which hop 1 (under-filled: 20–90 µs kernels) | ~0.4 ms | ~3% | 0.1–1 ms |
| per-minibatch start-up: host call, seed copy to device, state init, first launch | ~3–4 ms | ~20–25% | 1 ms |
| **GPU idle between kernels (25 device-wide boundaries)** | **~0.6–1.5 µs each, ~30 µs total** | **~0.2%** | **10 µs** |

Batch 64: 1.4 ms of GPU work against ~4.3 ms of start-up, so ~70% is fixed cost.
Orkut is similar (15 ms busy, ~3 ms start-up).

Memory system (microbenchmarks):

| array size | dependent load latency | independent random loads (throughput) |
|---|---|---|
| 64 KB | ~0.3 µs | ~2.4 G/s |
| 32 MB | ~0.3 µs | 0.35 G/s |
| 256 MB | **~1 µs** | **~0.09 G/s** |

The 256 MB throughput is about 90 loads in flight: the hardware cap even with 1 M
work-items in flight.

**Readings:**
- **Cross-SM synchronisation is not a cost here.** Back-to-back queued kernels leave
  ~1 µs gaps. Avoiding kernel boundaries (one minibatch per work-group, persistent kernels)
  would buy ~0.2%.
- **The fixed per-minibatch start-up (~3–4 ms) is the largest latency outside the real
  work.** That is what running K minibatches together hides.
- **The bulk (hop 3, ~2–3 M random accesses into large arrays) is bounded by the iGPU's
  random-access throughput (~0.1–0.35 G/s), not by occupancy.** More minibatches in
  flight cannot push past it on this GPU. This matches the measured K = 2/4 concurrency
  gains of +24% / +31% that then flatten.

CPU proxy for K = 4 per launch (one 4096-seed minibatch vs 1024; real batching would not
dedup across minibatches, so this is an upper estimate):
- seeds/s gain: +17% (products) / +6% (Orkut) at 6 threads, +54% / +43% at 14 threads.
- Even batched, 14 threads (~18 K seeds/s) stays below 6 threads (~24 K) and below pyg-lib
  with 14 workers (~47 K seeds/s, D20 sweep).
- So batching helps CPU scaling but does not replace multi-instance (D20).

**Not yet decomposed:** the 3–4 ms start-up, split between the host call, the pageable
seed copy, launching after an idle period, and the final readback. That is the next
measurement before designing.

## D22. Several minibatches per launch: "batched" vs "megakernel" prototypes — *measured, design open*
`cpp/include/gnnsample/multi.hpp` (`_C.MultiSampler`):
- **batched:** every no-sync kernel covers K minibatches, with padded per-minibatch
  segments and a segmented scan.
- **megakernel:** one `nd_range` kernel. Work-group m runs minibatch m end to end, with
  group barriers between phases.
- Both use one direct-mapped table per minibatch, label-early and sub-group sampling.
- Both are bit-identical per minibatch to `Sampler` and the reference (`tests/test_multi.py`,
  GPU and CPU, K = 1/3/8, work-groups of 256 and 1024).

GPU A/B (`bench/multi_ab.py`, batch 1024, [15,10,5], battery; logs `multi-ab-gpu.log`,
`multi-ab2-*.log`). Throughput relative to today's sampler in the same run:

| config | products | Orkut | LiveJournal |
|---|---|---|---|
| batched K=1 | 0.68 | 0.79 | 0.63 |
| batched K=4 | 1.00 | 1.15 | 1.12 |
| batched K=8 | 1.05 | 1.18 | 1.29 |
| megakernel K=8 (wg 512) | 0.65 | 0.74 | 1.04 |
| megakernel K=16 | 0.57 | 0.50 | **1.50** |
| megakernel K=32 | 0.57 | – | – |

Latency per call: today 8–23 ms; batched K=8 52–161 ms; megakernel K=16 87–612 ms.
Running all modes in one process exhausted Windows memory at Orkut (the guard killed it),
so each mode now runs in its own process.

**Readings (hypotheses until profiled):**
- **Batched:** K = 1 → 8 gives 1.5–2× its own throughput. The prototype pays 21–37%
  overhead at K = 1 (padded layout, per-item hop-state loads), so it ends 1.05–1.29× today.
  Removing that overhead is the obvious next step.
- **Megakernel on LiveJournal:** small minibatches (~270 K edges), where today's fixed
  start-up dominates (7.7 ms per minibatch). One work-group per minibatch has enough lanes;
  1.50× today.
- **Megakernel on products / Orkut:** heavier minibatches (~560–750 K edges). One work-group
  per minibatch is too narrow, and 16 minibatches' private data (~1 GB) exceeds the cache;
  0.50–0.65× today.
- **K = 16 → 32 adds nothing:** this iGPU's 64 compute units form 4 Xe-cores of 128
  hardware threads. Sixteen 512-wide work-groups already fill it.
- **Design direction:** choose the mapping per phase by its size. Per-work-group for small
  phases and small minibatches, whole-GPU kernels for large phases.

**Not yet done:** the CPU A/B of both modes, and profiler confirmation (VTune image built;
GPU metrics under WSL2 untested).

## D23. Locality without changing the sample: reordering, hub cache, cross-minibatch schedule — *measured: none kept as default (all kept as options)*
Motivation (D21): the sampler is bound by dependent random reads into the graph (~1 µs
each on the iGPU), not by compute, and the graph (~0.5 GB of neighbour lists on products)
does not shrink with the batch size. So the lever is *which cache lines* those reads land
on. All three changes leave every sampled minibatch bit-identical (nodes, local ids, edges,
CSC positions), so training cannot change. All are opt-in; today's `Sampler`, the batched
and the megakernel designs stay as they are, for the NVIDIA comparison later.

**1. Graph reordering** (`python/gnnsample/reorder.py`, `Sampler/MultiSampler.set_reorder`).
- Renumbered copy `data/csc/<g>@<method>`: same rows, neighbour order kept, ids mapped.
  Methods: `degree` (hubs first), `hubsort` (only hubs moved; Balaji & Lucia 2018), `rcm`
  (reverse Cuthill-McKee; too memory-hungry for Twitter), `random` (control).
- The sampler keys Philox on the original id (`order[v]`, read next to `indptr[v]`, so
  it adds bandwidth but not a dependent load), maps seeds in (host `perm`) and node ids and
  positions out (`export_to`, `sample`; positions need `pos_shift`). Features, labels and
  splits stay in original ids.
- Alternative: key Philox on the new id. Saves the `order[v]` read but changes the sample
  (still a valid sampler, but not bit-identical to PyG's/reference semantics), so rejected.
- Expected: fewer distinct lines per minibatch, mostly for `indptr` (8 nodes per line) and
  for hub rows. Social graphs have weak id locality to begin with; `random` tells how much
  the original numbering already gives.

**2. Hub cache** (`kHubHints` = 8192, `set_hubs(h)`, `hints.hpp`).
- On a hub-first numbering, the first h rows (h from a byte budget, `reorder.hub_count`)
  *are* the compact hub array: there is nothing left to copy. What remains is keeping them
  resident: non-hub neighbour reads are marked streaming (Intel LSC cache control, L1 + L3;
  other backends ignore it) so they stop evicting hub lines.
- Why hubs: a node is sampled as a neighbour in proportion to its out-degree, so frontiers
  are degree-biased and hub rows are expanded by almost every minibatch.
- Alternative: an explicit copy into shared local memory. Too small (64–128 KB per Xe-core)
  for neighbour lists, and it costs a load per work-group. Rejected for now.
- If the hint is wrong (cold lines were reused after all), it shows as a slowdown in the
  paired A/B; budgets 1/4/16 MB are swept.

**3. Cross-minibatch schedule** (`MultiSampler.set_schedule`, batched mode).
- Each hop, a counting sort over 2^16 node-id buckets orders the K frontiers' work items
  by node id, and the count and sampling kernels walk that order. The same node (and its id
  neighbours) expanded by several minibatches is read by neighbouring sub-groups at about
  the same time, so its lines come from cache after the first read. Each item still writes
  only its own minibatch's slots.
- Cost: 3 small kernels per hop (histogram, one-group scan, scatter) and a 4-byte work item
  per frontier node. Limits: k_max ≤ 255, K·frontier < 2^32.
- Not done: scheduling the edge-level dedup kernels by neighbour id (table locality); 10×
  more items, so measure the node-level schedule first.

**Measurements to run** (`tools/d23_runs.sh`; tests first):
- `bench/locality_stats.py`: hardware-independent counters per numbering: distinct
  indptr/neighbour lines per minibatch, reads per line, share of neighbour reads in the hub
  prefix (1/4/16 MB), and cross-K reuse (sum of per-minibatch lines / union lines, K =
  2..16). These project to NVIDIA caches without NVIDIA hardware.
- `bench/reorder_ab.py`: paired timing, today's sampler, original vs each numbering, ± hints.
- `bench/multi_ab.py --sched --hub-mb`: today / batched / batched+sched / megakernel on
  the original and the degree numbering.
- Keep a change only if the paired ratio is consistently below ~0.95 (rule 3).

**Results** (2026-10-25, battery 48–69%; ratios are paired within one process, so they
hold; absolute ms do not). Tests: 175 passed (`d23-test.log`). Logs `results/logs/d23-*.log`.

*Locality counters* (`locality_stats.py`, 32 minibatches; lines = distinct 64-byte lines per
minibatch):

| graph | indptr lines, original → best | neighbour lines (any numbering) | hub-prefix share (16 MB) | reuse over K=16: neighbour / indptr |
|---|---|---|---|---|
| products | 102K → 85K (rcm) | 466K | ≤1% | 1.74× / 5.3–6.2× |
| LiveJournal | 45K → 44K (rcm); degree 49K | 127K | ≤2% | 1.31× / 2.0–2.6× |
| Orkut | 119K → 112K | 505K | ≤2% | 1.31× / 4.9–5.3× |

Neighbour-list lines are 75–85% of graph lines and no numbering moves them: Floyd picks
random positions inside each list. Hub rows draw ≤2% of reads, because dedup expands a hub
at most once per minibatch and reads only k of its ~10^4 neighbours.

*Timing, today's sampler* (`reorder_ab.py`, paired median ratio vs the original numbering):

| numbering | products | LiveJournal | Orkut |
|---|---|---|---|
| degree | 0.978 (IQR 0.964–0.994) | 1.050 | 1.139 |
| hubsort | 0.996 | 1.067 | – |
| rcm | 1.065 | 1.146 | 1.148 |
| random | 1.206 | 1.315 | – |
| + HUB_HINTS (1/4/16 MB) | within ±0.01 of its numbering | within ±0.03 | 0.993× of degree |

*Multi-minibatch* (`multi_ab.py`, throughput vs today in the same process):

| config | products | products@degree | LiveJournal | LJ@degree | Orkut | Orkut@degree |
|---|---|---|---|---|---|---|
| batched K=8 | 1.10 | 1.17 | 1.26 | 1.12 | 0.99 | 1.01 |
| batched+sched K=4 / 8 / 16 | 0.84 / 0.81 / 0.67 | 0.87 / 0.88 / 0.75 | 0.91 / 1.59 / 1.76 | 0.90 / 0.99 / 1.05 | 0.88 / 0.85 / 0.69 | 0.96 / 0.87 / 0.88 |
| megakernel K=16 | 0.44 | 0.56 | 0.64 | 1.20 | 0.44 | 0.75 |

**Readings:**
- **Reordering:** no numbering clears the 0.95 bar. Only products@degree is faster (2%).
  The original numberings already carry locality (random is 21–32% slower), and the
  reordered paths pay one extra random read per expanded node (`order[v]` for the Philox
  id), about as many lines as the indptr saving. That likely explains why rcm, with the
  fewest indptr lines, is slower. Not yet isolated; see next steps.
- **Hub cache hints:** no effect (±1–3%, inside the IQR). This matches the counters:
  ≤2% of reads are hub reads.
- **Locality schedule:** slower in 14 of 15 cells. Its 3 extra kernels per hop cost more
  than 1.3–1.7× neighbour-line reuse returns. The one win (LiveJournal original, K=8/16:
  1.59/1.76) came from the run whose "today" baseline was 2.3× slower than in every other
  LiveJournal run (13.9 vs 5.5–7.2 ms per call), so it needs a rerun before anyone believes it.
- **Batched K=8** reproduces D22 (1.0–1.26×). **Megakernel K=16** varies widely between
  runs (LiveJournal 0.64 vs 1.20; D22 1.50), so it is not settled on this GPU.
- **Preprocessing cost** (Python/NumPy, one-off): 10–38 s per graph and method. With
  products@degree's 2% (~0.25 ms × 193 minibatches ≈ 0.05 s per epoch), break-even is about
  400 epochs.

**Decision:** none of the three becomes a default. All stay as tested options (bit-identical,
175 tests), and the three sampler designs stay side by side for the NVIDIA comparison.
The counters say the remaining graph cost is random reads inside neighbour lists, which
no layout fixes. The levers left are latency hiding (more reads in flight per thread) and
fewer reads, not cache placement.

**Next (not done):** (1) a layout-only run, keying Philox on the new ids for the
measurement only, to separate the `order[v]` cost from the layout effect; (2) rerun
LiveJournal batched+sched on AC to confirm or kill the 1.76×.

**Status (user decision, 2026-10-25):** reordering (`reorder.py`, `set_reorder`), hub cache
hints (`HUB_HINTS`, `set_hubs`) and the cross-minibatch schedule (`set_schedule`) **stay in
the codebase as tested, non-default options**. They are not part of the primary pipeline.
**Retest all three on an NVIDIA GPU** with enough memory and SMs. Larger L2 (40–96 MB),
100+ SMs and more reads in flight change the cache and scheduling trade-offs measured here
on a 4-Xe-core iGPU, so these results do not carry over. The retest uses the same scripts:
`tools/d23_runs.sh` (locality counters, `reorder_ab.py`, `multi_ab.py --sched --hub-mb`).

## D24. End-to-end training: batched (K=8) vs today's sampler — *measured: tie on the iGPU, today's stays primary*
User rule: adopt batched as the primary method only if it beats today's pipeline in
end-to-end training. Setup (`bench/e2e_batched_ab.py`, AC power):
- **Configurations:** four, in one process with one shared feature table: today serial,
  batched serial, today overlapped (`PrefetchingGpuSampler`, production), batched
  overlapped (`BatchedGpuSampler`, new).
- **Fair comparison:** every configuration trains on bit-identical minibatches (same
  seeds and keys), one full epoch per round, model re-initialised with the same seed, and
  the configuration order rotates each round.
- **Start-up included:** epoch time includes the wait for the first minibatch (batched
  waits for its whole first block); the start-up wait is also reported on its own.
- **Model:** GraphSAGE 3×256, [15,10,5], batch 1024, trimmed layers.
- **New code:** `BatchedGpuSampler` samples K minibatches per launch, exports their graphs
  into 2×K slots, and gathers features per minibatch just before use (2 feature slots, so
  no 2.6 GB of resident features). Slot reuse is guarded by semaphores plus XPU events.
  Bit-identical to the serial path (`tests/test_pipeline.py`, 6 new cases).

Results, mean over 3 rounds (ms per step unless noted; logs `results/logs/d24-e2e-*.log`):

| graph | config | epoch s | vs today serial | start-up ms | wait | train | sample |
|---|---|---|---|---|---|---|---|
| products (192 mb) | today serial | 58.9 | 1.000 | 203 | 44.3 | 262.5 | 20.1 |
| | batched serial | 56.6 | 0.962 | 151 | 38.9 | 255.8 | 15.7 |
| | today overlapped | 55.9 | 0.958 | 40 | 18.1 | 273.2 | – |
| | batched overlapped | 55.8 | 0.953 | 145 | 4.8 | 286.0 | – |
| LiveJournal (47 mb) | today serial | 5.43 | 1.000 | 19 | 18.8 | 96.7 | 8.6 |
| | batched serial | 5.33 | 0.985 | 72 | 17.2 | 96.2 | 6.8 |
| | today overlapped | 5.31 | 0.984 | 20 | 5.1 | 107.9 | – |
| | batched overlapped | 5.36 | 0.993 | 74 | 3.3 | 110.7 | – |

Products: per-round paired ratios are batched/today serial 0.87, 1.03, 0.99 and
overlapped 1.02, 0.94, 1.04. Run-to-run spread of one configuration is ±10%.
Losses match today serial to ≤ 9e-4 (identical inputs; PyTorch's float atomics). Products
gathers take ~21 ms per step in every configuration.

**Readings:**
- **Sampling** is 21–22% faster per minibatch with batched (products 20.1 → 15.7 ms,
  LiveJournal 8.6 → 6.8 ms), as in D22.
- **The epoch does not move beyond noise.** Training is 85–90% of a step on this iGPU
  (256–286 ms on products), so ~4 ms saved per step is ≤1.5%, under the ±10% spread.
- **Overlap** hides the data wait (wait → 3–18 ms), but training steps get 5–12% slower
  because sampling shares the 4 Xe-cores with training. Batched overlapped ties today
  overlapped (55.8 vs 55.9 s).
- **Start-up:** batched waits for a whole block before the first step (72–151 ms vs
  19–40 ms overlapped). Small per epoch (≤0.3%), but real latency.
- **Orkut could not be measured.** The paired setup holds two device copies of the graph
  (0.94 GB each) and was killed by the low-memory guard three times (logs `*-killed`,
  `*-dockereof`, `*-lowmem`; one partial run's epochs swung 12–23 s from paging).

**Decision:** by the agreed rule, batched does not beat today's pipeline end to end here,
so **today's sampler stays primary**. `BatchedGpuSampler` stays as a tested option. On a
discrete NVIDIA GPU, training steps are far shorter and sampling is a larger share of each
step, so rerun this benchmark there (Phase 4b) before deciding finally.

**If revisited on this laptop:** let MultiSampler share PySampler's device graph (one
copy instead of two) so that Orkut fits.

**Direction (user decision, 2026-10-25): batched (K minibatches per launch) is the primary
sampling design for further optimisation work**, ahead of today's one-minibatch pipeline.
The decision rests on the expected behaviour on GPUs with more resources, not on the
iGPU end-to-end result above (a tie). Rationale:
- The iGPU tie is a resource limit. Training takes 85–90% of each step, and overlap
  slows training because sampling and training share 4 Xe-cores.
- With more SMs and faster training steps, sampling becomes a larger share of the step
  and contention drops, so batched's 21–22% sampling gain should reach the epoch time.
- To confirm on NVIDIA (Phase 4b), with `bench/e2e_batched_ab.py` unchanged.

Points to handle while it becomes primary:
- **Start-up wait:** paid at every epoch start (the first block has to finish), not once
  per run. Removable by starting the next epoch's first block during the previous epoch's
  last steps.
- **Memory:** about 2 GB on products at K=8 (graph copy, K-minibatch buffers, slots).
  Sharing the device graph with the single-minibatch sampler removes one copy.
- **Not yet batched:** the drop-in `NeighborLoader` and `GpuSampler` /
  `PrefetchingGpuSampler` (the CPU path included) still use today's sampler. Today's
  sampler and the megakernel stay in the code for comparison (D22/D23).

**Switched (2026-10-25):** the batched sampler is now the default everywhere training happens:
- `gnnsample.NeighborLoader` (`sampler="batched"`, `K=8`), `loader.training_sampler()`,
  `bench/accuracy.py`, and `train_e2e.py --config ours-batched` (added to
  `tools/bench_suite.sh`).
- `sampler="single"` keeps the earlier pipeline. `NeighborLoader` falls back to it, with a
  warning, for custom options or fanouts above 16 (MultiSampler's limit).
- The batched sampler gained a CPU path, edge positions (`e_id`), and fresh tensors when
  `prefetch=False`.
- Batches are bit-identical to the single sampler's: `tests/test_neighbor_loader.py` runs
  every loader test with both samplers, and compares whole epochs (XPU/CPU, with and without
  prefetch, partial last block).
- Benchmarks that measure a specific path (`overlap_ab.py`, `sampler_stages.py`,
  `profile_pyg.py`, `train_e2e.py` ours / ours-overlap) keep their explicit classes.
- Bug found by the switch's tests and fixed: an iteration that ended without resuming the
  generator after its last minibatch (NeighborLoader's `zip(batches, results)`), or that
  stopped early with `break`, left slots held, and the next epoch waited forever
  (`results/logs/d24-hang-nl.log`). Now every exit path stops the producer and releases
  every slot, including those of prepared but unconsumed minibatches. A regression test
  covers both cases. Full suite: 198 passed (`results/logs/d24-switch-test.log`).

## D25 (memory). Batched sampler footprint: shared graph, hash dedup tables, slots sized to need — *measured: 44–61% less; kept*
Motivation: the batched path (primary, D24) holds about 2 GB on products at K=8, and the
D24 Orkut run was killed by the low-memory guard (two device copies of a 0.94 GB graph).
Three changes, each output-preserving (bit-identical to `reference.py`) and each
switchable back to the D24 behaviour:

1. **Shared device graph.** `_C.MultiSampler.from_sampler(sampler, fanouts, max_batch,
   k_max)` runs on a `_C.Sampler`'s device graph instead of copying it. Its queue uses that
   sampler's device and context (the platform default context, as `share_features` does);
   nanobind `keep_alive` keeps the sampler alive. `BatchedGpuSampler(...,
   share_graph_with=<GpuSampler>)` uses it (not for reordered graphs);
   `bench/e2e_batched_ab.py` now does. Off: pass `indptr, indices` instead.
2. **Hash dedup tables.** `MultiSampler.set_table("auto"|"direct"|"hash")`. Direct tables
   cost 8 B per *graph* node per minibatch. The hash option is `GenHashView` (16-byte
   generation-tagged slots) with the smallest power of two ≥ `max_nodes` slots per
   minibatch, which is the active half of `Sampler`'s `kSmallTable` table. It never fills:
   a minibatch inserts at most `max_nodes` distinct keys per generation, and stale slots are
   reclaimable, so a new key always finds a slot. `auto` (default) picks the smaller table;
   ties go to direct. All K tables share one 8-bit generation and are fully reset at first
   use, after `set_table`, and when the generation wraps (every 255 runs). The batched and
   megakernel kernels are templated over the view (seeds: `s = insert(v); set_val(s, i)`).
   The 2^24 limits are unchanged.
3. **Slots sized to need** (`BatchedGpuSampler(slots="grow")`, default; `"max"` = D24).
   Graph slots (2 × K × (nodes + 2·edges) int64) and feature slots (2 × nodes × dim float32)
   are no longer allocated for the worst case at construction. The first block that uses
   them sets a target of its largest minibatch × 1.25 (capped at the worst case). A slot
   that a later minibatch does not fit is reallocated at max(target, need × 1.25), and the
   target follows. Reallocation happens only where the producer holds the slot (semaphore
   taken, the consumer's XPU event synchronised in `_take`), so no queued kernel reads the
   old storage. A view the consumer still holds keeps the old storage alive. After any
   allocation, `torch.xpu.current_stream().synchronize()` runs before our queue writes.
   `overlap=False` allocates no slots at all.

Accounting: `MultiSampler.device_bytes()` / `memory()` (graph only if owned, tables,
buffers) and `BatchedGpuSampler.device_bytes()` / `memory()` (plus slots);
`bench/batched_memory.py` prints old vs new by part.

**Expected bytes** (computed from the allocation sizes; K=8, batch 1024, [15,10,5], so
`max_nodes` = 937,984 on every graph below, `max_edges` = 936,960, 2^20 hash slots = 16 MiB
per minibatch; MiB):

| part | products | LiveJournal | Orkut | Twitter-2010 |
|---|---|---|---|---|
| nodes | 2,449,029 | 4,847,571 | 3,072,441 | 41,652,230 |
| graph copy, old → new | 490.6 → 0 | 298.2 → 0 | 917.5 → 0 | 5,919.1 → 0 |
| dedup tables, direct → auto (hash) | 149.5 → 128.0 | 295.9 → 128.0 | 187.5 → 128.0 | 2,542.3 → 128.0 |
| per-minibatch buffers (unchanged) | 314.7 | 314.7 | 314.7 | 314.7 |
| graph slots, worst case (old) | 343.2 | 343.2 | 343.2 | 343.2 |
| feature slots, worst case (old, dim 100) | 715.6 | 715.6 | 715.6 | 715.6 |
| **old total** | **2,013.7** | **1,967.7** | **2,478.6** | **9,835.0** |
| **saved by 1 + 2 alone** | **512.1** | **466.1** | **977.0** | **8,333.4** |

- Slots ("grow") depend on the minibatches seen, so they are not in the "saved" row.
  Products, from the D9 means (409 K nodes, 715 K edges) × 1.25: graph slots ≈ 281 (−62)
  and feature slots ≈ 390 (−326). The first block's *largest* minibatch sets the size, so
  real savings are somewhat smaller. Edge slots barely shrink on products (715 K × 1.25 is
  close to the 937 K worst case); node and feature slots do.
- Twitter-2010: the old configuration needs the 5.9 GB graph twice plus 2.5 GB of tables,
  which does not fit; the new one adds 443 MiB (tables + buffers) plus its slots to the
  sampler's graph.
- Products at K=8 old: ~2.0 GB, which matches D24's estimate.

**Timing: not yet measured.** On products `auto` now picks hash (128 vs 149.5 MiB), which
probes and claims with a CAS where direct does not (D18 measured direct as slightly
faster on today's sampler). Rule 3 applies: paired A/B of `set_table("direct")` vs
`"hash"` in batched mode on products, LiveJournal and Orkut before relying on `auto`'s
choice for speed. The memory numbers above are to be confirmed with
`bench/batched_memory.py` (and torch's `memory_allocated` delta as a cross-check).

**Measured (2026-10-25, battery; `bench/batched_memory.py`, logs `results/logs/d25-mem-*.log`).**
MB of the batched sampler at K=8. Slot sizes are confirmed by torch's allocation delta,
and no slot had to grow:

| graph | old | new | saved | biggest saving |
|---|---|---|---|---|
| products | 2,014 | 1,122 | 892 (−44%) | graph copy 491, feature slots 321 |
| LiveJournal | 1,968 | 759 | 1,209 (−61%) | feature slots 520, graph 298, slots 223, tables 168 |
| Orkut | 2,479 | 1,292 | 1,187 (−48%) | graph copy 918 |
| Twitter-2010 | does not fit | 637 | – | tables 128 instead of 2,542; graph shared |

The remaining fixed cost is the per-minibatch buffers (315 MB, sized to the worst case
937 K nodes/edges per minibatch). Real minibatches are 2–7× smaller (products max 416 K
nodes, Twitter 131 K), so those buffers are the next memory target.

**Table timing** (`multi_ab.py --tables 8`, paired in one process, 3 rounds; throughput vs
today's sampler):

| graph | direct ÷ hash bytes | batched direct | batched hash |
|---|---|---|---|
| products | 1.17× | **1.41** | 1.31 |
| Orkut | 1.46× | 1.01 | 1.04 |
| LiveJournal | 2.3× | 1.02 | **1.25** |

`auto` was changed to choose hash only when the direct table is at least 1.5× larger.
Products returns to direct (no probing, as in D18); LiveJournal and Twitter use hash
(LiveJournal's hash table packs its touched entries into fewer lines). Single runs on
battery: re-check on AC before treating the 7% and 23% as exact.

**Bug found and fixed while testing:** the serial / CPU batched path resized a feature
slot that does not exist (`TypeError`, 14 tests). 231 tests pass
(`results/logs/d25-test.log`). One earlier full run failed while Windows had under 1 GB
free: device allocations failed under memory pressure. The tests pass alone and in the
full suite once memory is free (`d25-test-lowmem.log`).

**Orkut end-to-end (D24 retry):** still killed by the low-memory guard, in round 0. The
batched side is now 464 MB plus slots. The rest is today's pipeline in the same process
(its own worst-case slots, ~0.8 GB, and the 0.94 GB graph), host features (1.2 GB) and
training, on a laptop whose free memory varied between 0.2 and 9 GB during the session.
Not retried further here.

## D25 (profile). Where the batched sampler's time goes — *measured*
`bench/multi_profile.py` (every kernel timed on a profiling queue, no barriers added;
battery; logs `results/logs/d25-profile-*.log`). Milliseconds per minibatch:

| | single (today) | batched K=1 | batched K=8 | K=8: GPU busy / idle / host |
|---|---|---|---|---|
| products | 29.0 | 22.5 | **17.9** (−38%) | 16.6 / 0.6 / 0.6 |
| LiveJournal | 11.7 | 14.6 | **8.6** (−26%) | 7.3 / 0.6 / 0.7 |
| Orkut | 27.7 | 23.3 | **20.9** (−25%) | 19.7 / 0.5 / 0.7 |

- **Batching removed the overhead.** At K=8 the GPU computes 93–94% of the time; launch
  gaps and host time are ~1.2 ms per minibatch together. Further gains must make the
  kernels themselves faster.
- **Hop 3 is 75% of the time** (products 12.7 of 16.6 ms; Orkut 15.6 of 19.7 ms).
- **Deduplication, not sampling, is the largest cost: 60–65% of GPU time.** Products K=8:
  insert 3.5, assign 3.3, flag 1.9, relabel 1.0, scan-flag 0.8 ms (10.5 ms) vs sample
  5.5 ms. All of it is random access into the per-minibatch table. Next candidates: fuse
  insert+flag, cheaper last-hop dedup (its new nodes are leaves), and table locality
  (see the hash-vs-direct result above).

**VTune 2026.6 under WSL2** (`docker compose run vtune`, image `gnnsample-vtune`):
- `xpu-offload` and `gpu-hotspots` collect nothing (elapsed 0 s, no GPU section). VTune
  cannot see the iGPU behind WSL2's paravirtual `/dev/dxg`; GPU collectors need the
  native Linux GPU driver and its metrics library.
- `hotspots` (CPU, user-mode sampling) works, but the result is dominated by VTune's own
  instrumentation and start-up (JIT, loading); the sampler's host side is ~0.7 ms per
  minibatch.
- `vtune-self-checker.sh` ends with "unknown failure".
- GPU hardware counters (cache hit rates, stall reasons) need native Linux, or VTune
  installed natively on Windows (a system install), or Nsight Compute on the NVIDIA server.
