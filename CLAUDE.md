# gnnsample: agent quick-start

gnnsample is a SYCL/DPC++ GPU neighbour sampler and feature gather for GraphSAGE
training. It uses fanout [15,10,5] and is benchmarked against PyG/pyg-lib on an Intel
laptop iGPU, inside Docker on WSL2.

- **Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first.** It covers components,
  data flow, the per-hop kernels, option bits and invariants.
- **Primary sampling design (D24, user decision): batched, K minibatches per launch
  (`MultiSampler` batched mode, `BatchedGpuSampler`).** New optimisations target it;
  today's single-minibatch `Sampler` and the megakernel stay for comparison.
- **Read [`docs/decisions.md`](docs/decisions.md) before changing a design choice.** It
  records why each choice was made and holds every measured number.

This is a code-only copy: datasets, results and caches are not
included. Rebuild everything with [`docs/PROCEDURE.md`](docs/PROCEDURE.md).

## Commands

All commands run inside the `gnnsample-dev` container, started from the project root.

**Docker on Windows:**

- If the Docker CLI is not on PATH, add your Docker Desktop `resources\bin` folder to it
  (the `tools/*.sh` scripts have a commented line for this).
- In Git Bash, prefix commands with `MSYS_NO_PATHCONV=1`.

Build the image (needs `gridguard:latest`, built from the sibling `GridGuard` folder):

```bash
docker compose build
```

Configure and build the extension (writes `python/gnnsample/_C*.so`):

```bash
docker compose run --rm -T dev cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
```

```bash
docker compose run --rm -T dev cmake --build build
```

Tests (72, about 1 minute; they need the iGPU):

```bash
sh tools/run_logged.sh test 1800 python -m pytest -q tests/
```

```bash
docker compose run --rm -T dev python -m pytest --collect-only -q tests/
```

Benchmarks (long; always through `run_logged.sh`, on AC power):

```bash
sh tools/run_logged.sh ablate-x 2400 python bench/ablate.py --graph products --variants 8169,0
```

```bash
sh tools/run_logged.sh e2e-ours 1800 python bench/train_e2e.py --config ours-overlap
```

Watch long runs for stalls (prints STALL / DONE / GONE):

```bash
sh tools/watch_runs.sh 300
```

Prepare datasets (products, livejournal, orkut):

```bash
docker compose run --rm -T dev sh data/prepare_small.sh
```

## Rules

1. **Bit-identity is the correctness bar.** Every sampler path and every option
   combination must match `python/gnnsample/reference.py` exactly: nodes, local ids,
   edges and their order, and per-hop counts. Never change the reference to make the GPU
   code pass.
2. **Every optimization is a new run-time option bit** (`SamplerOption` in
   `cpp/include/gnnsample/sampler.hpp`). Keep `options = 0` (v1) and the existing bits
   working. When you add a bit, do all of the following:
   - expose it in `cpp/src/module.cpp`;
   - add it to `NAMES` in `bench/ablate.py`;
   - add masks to `ALL_OPTIONS` and to the switching test in `tests/test_gpu_sampler.py`.

   Respect the invariants in ARCHITECTURE.md §5.1: label-early implies assign-from-scan;
   tagged tables need ids below 2²⁴; generations wrap at 255; `node_slot_` validity;
   empty-batch state init.
3. **Back claims with paired measurements.** Use `bench/ablate.py`, which interleaves
   variants in one process, and report the median per-batch ratio on products,
   LiveJournal and Orkut. Separate before/after runs are not valid evidence, because the
   iGPU drifts up to 2×. Keep a bit only if it is consistently below about 0.95. Record
   the result as a new `D<n>` entry in `docs/decisions.md`, with its reason, alternative,
   numbers and log path. Add the bit to `DEFAULT_OPTIONS` in
   `python/gnnsample/loader.py` only if it is kept.
4. **Long jobs** go through `tools/run_logged.sh <name> <timeout_s> <cmd>`. That gives a
   hard timeout, a log in `results/logs/` and the power state. Run `tools/watch_runs.sh`
   at a 5 to 10 minute interval. Quote absolute timings only from AC-power logs.
5. **The GPU is shared.** Check for running `gnnsample-*` containers before benchmarking.
   Never run two GPU benchmarks at once.
6. **Never edit `results/` or `data/csc/` by hand.** Never edit `.gpu_cache/` either; it is
   the persistent kernel JIT cache, and rebuilding it takes about 10 minutes.
7. **Keep the toolchain pins.**
   - DPC++ must stay at **2026.1.0** to match the SYCL runtime that PyTorch 2.14.1+xpu
     bundles. The extension and torch share one runtime and one default SYCL context.
   - Feature tables are chunked at 512 MB because of the iGPU's 1.07 GB allocation cap.
   - Feature dim must be a multiple of 4.
   - Seeds must be unique.
   - The queue must be in-order.
