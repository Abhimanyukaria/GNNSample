# gnnsample

**gnnsample** is a header-only **SYCL/DPC++ GPU neighbor sampler and zero-copy feature gather** for GraphSAGE-style minibatch GNN training, bound to Python with `nanobind`. It provides a drop-in replacement for PyTorch Geometric's (`torch_geometric`) `NeighborLoader` on homogeneous graphs.

Developed and benchmarked on an Intel Core Ultra 7 265U laptop iGPU (inside Docker on WSL2) against PyG 2.8.1 + `pyg-lib` 0.9.0, `gnnsample` achieves:
- **6.4–7.9× faster** per-minibatch neighbor sampling than `pyg-lib`.
- **1.35× faster** mean end-to-end training step time than PyG's best multiprocess worker setting.
- **44–61% lower device memory footprint** in batched multi-minibatch mode (`K=8`) via shared device graphs, generation-tagged hash/direct deduplication tables, and dynamic slot sizing.
- **Bit-identical determinism** against a NumPy reference implementation (`python/gnnsample/reference.py`) across all sampler modes, runtime option flags, and SYCL devices, with full test-accuracy parity on `ogbn-products` ($p = 0.90$).

---

## Key Features

- **Drop-In PyG Compatibility (`gnnsample.NeighborLoader`)**: Swap one import line in a standard PyTorch Geometric training loop. Yields `Data` batches already on the XPU with `num_sampled_nodes` and `num_sampled_edges` populated for hierarchical layer trimming (`trim_to_layer`).
- **Zero-Sync & Multi-Batch GPU Pipeline (`Sampler` & `MultiSampler`)**:
  - **No-Sync Execution**: Upper-bound kernel launches with device-side `HopState` tracking and device-sized exclusive prefix scans (`exclusive_scan_dev`), requiring only **1 host readback per launch** instead of per-hop host synchronization.
  - **Batched Mode (`K` minibatches per launch)**: Amortizes fixed launch and start-up overhead across $K=8$ minibatches simultaneously, keeping the GPU busy >93% of the time.
  - **Sub-Group Floyd Sampling**: Assigns one 16-lane sub-group per frontier node (`k <= 16`), replaying Robert Floyd's without-replacement sampling algorithm in parallel via sub-group shuffles while coalescing small-degree neighbor copies.
  - **Generation-Tagged Deduplication (`DirectView` & `GenHashView`)**: Eliminates per-batch 32 MB table resets using 8-bit generation tags (`1..255`) and auto-selects between direct-mapped tables ($8\text{ B} \times N$) and power-of-two open-addressing hash tables ($16\text{ B} \times \text{cap}$).
- **Zero-Copy Host-USM Feature Gather (`FeatureTable`)**: Stores large node feature matrices in 512 MB chunked host USM (bypassing the iGPU's 1.07 GB single-allocation cap) and gathers directly into PyTorch XPU tensors using vectorized `float4` loads on a shared SYCL context.
- **Double-Buffered Prefetching**: Overlaps next-batch sampling and feature gathering with current-batch PyTorch training using a background worker thread and `torch.xpu.Event` slot synchronization.
- **Counter-Based Determinism**: Keyed Philox-4x32-10 RNG `(seed, hop, node, draw)` combined with atomic-min first-edge discovery + exclusive scan ensures local node IDs and edge orderings are independent of thread scheduling.

---

## Quick Start (Drop-In PyG Usage)

Switching an existing PyTorch Geometric training script to `gnnsample` requires changing only the import:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GraphSAGE

# Drop-in replacement for: from torch_geometric.loader import NeighborLoader
from gnnsample import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[15, 10, 5],
    input_nodes=train_idx,
    batch_size=1024,
    shuffle=True,
)

device = torch.device("xpu")
model = GraphSAGE(data.num_features, 256, num_layers=3, out_channels=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in loader:
    batch = batch.to(device)  # No-op for gnnsample: tensors are already on XPU
    optimizer.zero_grad()
    out = model(
        batch.x,
        batch.edge_index,
        num_sampled_nodes_per_hop=batch.num_sampled_nodes,
        num_sampled_edges_per_hop=batch.num_sampled_edges,
    )[: batch.batch_size]
    loss = F.cross_entropy(out, batch.y[: batch.batch_size])
    loss.backward()
    optimizer.step()
```

See [`examples/train_products.py`](examples/train_products.py) for a runnable side-by-side comparison (`--loader gnnsample` vs `--loader pyg`).

---

## Repository Structure

```text
├── CMakeLists.txt              # Builds the nanobind _C extension and SYCL microbenchmark tools
├── docker-compose.yml          # Container config with WSL2 iGPU passthrough (/dev/dxg) & 4GB shm
├── docker/
│   ├── Dockerfile              # gnnsample-dev image (Intel DPC++ 2026.1 + PyTorch XPU + PyG)
│   ├── vtune/Dockerfile        # Optional profiler image with Intel VTune
│   └── webgraph/               # Java WebGraph decoder for streaming large BVGraphs (Twitter-2010) to CSC
├── cpp/
│   ├── include/gnnsample/
│   │   ├── sampler.hpp         # Single-minibatch multi-hop SYCL sampler, dedup views & option bits
│   │   ├── multi.hpp           # MultiSampler: batched (K minibatches/launch) & megakernel modes
│   │   ├── gather.hpp          # Chunked USM FeatureTable and vectorized float4 gather kernel
│   │   ├── scan.hpp            # Device-wide exclusive prefix sum (ScanWorkspace)
│   │   ├── philox.hpp          # Stateless Philox-4x32-10 RNG and Lemire multiply-shift bounded draw
│   │   └── hints.hpp           # Intel LSC streaming cache hints for non-hub neighbor reads
│   ├── src/module.cpp          # nanobind Python bindings (_C.Sampler, _C.MultiSampler, _C.Gather)
│   └── tools/                  # Standalone SYCL utilities (device_info, launch_overhead)
├── python/gnnsample/
│   ├── neighbor_loader.py      # Drop-in PyG NeighborLoader replacement
│   ├── loader.py               # GpuSampler, PrefetchingGpuSampler, BatchedGpuSampler
│   ├── reference.py            # NumPy reference sampler (ground-truth semantics definition)
│   ├── philox.py               # Vectorized NumPy Philox-4x32-10 matching philox.hpp
│   ├── models.py               # GraphSAGE with hierarchical layer trimming (layer_adjs)
│   ├── graph.py                # Binary CSC graph loader
│   └── reorder.py              # Graph reordering (degree, hubsort, RCM, random)
├── data/                       # Dataset downloaders and binary CSC converters (products, LJ, Orkut, Twitter)
├── bench/                      # Paired interleaved A/B benchmarks, stage profilers, e2e training & accuracy
├── tests/                      # pytest suite verifying bit-identity across all options, modes & edge cases
├── tools/                      # Logged execution wrapper (run_logged.sh), stall watcher & benchmark suites
└── docs/
    ├── ARCHITECTURE.md         # Deep-dive into data flow, per-hop kernels, invariants & USM layout
    ├── PROCEDURE.md            # Step-by-step reproduction guide from scratch
    ├── decisions.md            # Decision log (D1–D25) with hypotheses, alternatives & measured numbers
    ├── research.md             # Literature review, baselines & dataset characteristics
    └── report.html             # Interactive visual benchmark report
```

---

## Build & Reproduction

All builds, tests, and benchmarks run inside the `gnnsample-dev` Docker container. For full step-by-step instructions, see [`docs/PROCEDURE.md`](docs/PROCEDURE.md).

### 1. Build the Container & Extension

```bash
docker compose build
docker compose run --rm -T dev cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
docker compose run --rm -T dev cmake --build build
```

*(Note: On Windows Git Bash, prefix `docker compose run` commands with `MSYS_NO_PATHCONV=1`.)*

### 2. Prepare Datasets

Downloads `ogbn-products`, `soc-LiveJournal1`, and `com-Orkut` into `data/raw/` and converts them into the compact binary CSC format (`indptr.int64.bin`, `indices.int32.bin`, `meta.json`) under `data/csc/<name>/`:

```bash
docker compose run --rm -T dev sh data/prepare_small.sh
```

### 3. Run the Correctness Test Suite

Every sampler path (`Sampler`, `MultiSampler` batched/megakernel, CPU/XPU, graph reordering, and all `SamplerOption` bit combinations) is tested for exact bit-identity against `python/gnnsample/reference.py`:

```bash
sh tools/run_logged.sh test 1800 python -m pytest -q tests/
```

### 4. Run Benchmarks

Long jobs are executed via `tools/run_logged.sh` (which enforces timeouts, records AC/battery power state, and logs to `results/logs/`):

```bash
# Paired, interleaved per-option ablation on ogbn-products
sh tools/run_logged.sh ablate-products 2400 python bench/ablate.py --graph products --variants 8169,0

# End-to-end GraphSAGE training step benchmark
sh tools/run_logged.sh e2e-ours 1800 python bench/train_e2e.py --config ours-batched

# Full benchmark suite across datasets
sh tools/bench_suite.sh
```

---

## Documentation

- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**: Complete technical reference covering the CSC data model, per-hop kernel pipeline, deduplication table views, runtime option bitmask (`SamplerOption`), and invariants.
- **[`docs/PROCEDURE.md`](docs/PROCEDURE.md)**: End-to-end reproduction commands.
- **[`docs/decisions.md`](docs/decisions.md)**: Engineering decision records (`D1`–`D25`) documenting every architectural choice, rejected alternative, and paired A/B measurement.
