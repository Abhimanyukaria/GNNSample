"""Sampling at scale (Twitter-2010 and smaller graphs) within the laptop's 10 GB.

Twitter's CSC is 6.2 GB, so a NumPy copy plus a GPU copy would not fit. Each sampler runs in
its own process on the same seed batches (a random 1% of nodes, gSampler's convention;
batch 1024, fanout [15,10,5]):
  --mode pyg    pyg-lib with the CSC in RAM as int32 (it accepts int32; 1.47 B edges < 2^31)
  --mode ours   our sampler; the CSC is memory-mapped and streamed into GPU memory, so only
                the device copy stays resident. v1 (options 0) and the final defaults are
                timed interleaved per batch (paired), then GPU stage times for the defaults.
Appends one JSON line per run to results/scale.jsonl.

Usage: python bench/scale.py --graph twitter --mode pyg|ours [--batches 40]
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from gnnsample.graph import load

FANOUTS = [15, 10, 5]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def seed_batches(g, n, batch):
    rng = np.random.default_rng(0)
    seeds = rng.choice(g.num_nodes, int(0.01 * g.num_nodes), replace=False)
    seeds = rng.permutation(seeds)
    n = min(n, len(seeds) // batch)
    return [seeds[i * batch:(i + 1) * batch].astype(np.int32) for i in range(n)]


def run_pyg(g, batches, threads):
    import torch
    from pyg_lib.sampler import neighbor_sample

    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    rowptr = torch.from_numpy(np.asarray(g.indptr).astype(np.int32))
    col = torch.from_numpy(np.fromfile(Path(g.indices.filename), dtype=np.int32))
    load_s = time.perf_counter() - t0
    times, edges = [], []
    for b in batches:
        t0 = time.perf_counter()
        out = neighbor_sample(rowptr, col, torch.from_numpy(b), FANOUTS, csc=True)
        times.append(time.perf_counter() - t0)
        edges.append(out[0].numel())
    return dict(load_s=load_s, median_ms=float(np.median(times[2:]) * 1e3),
                mean_edges=float(np.mean(edges)), dtype="int32")


def run_ours(g, batches):
    from gnnsample import _C
    from gnnsample.loader import DEFAULT_OPTIONS

    t0 = time.perf_counter()
    d = Path(g.indices.filename).parent
    paths = (str(d / "indptr.int64.bin"), str(d / "indices.int32.bin"))
    # Chunked file -> device load: keeps the VM's page cache small (device memory is
    # outside the WSL VM on the iGPU; see D19).
    s = _C.Sampler.from_files(*paths, FANOUTS, 1024, "gpu")
    load_s = time.perf_counter() - t0
    variants = [0, DEFAULT_OPTIONS]
    for v in variants:  # warm-up: JIT and first table initialisation per variant
        s.set_options(v)
        s.run(batches[0], 0)
        s.run(batches[1], 1)
    times = {v: [] for v in variants}
    edges = []
    for i, b in enumerate(batches):
        for v in (variants if i % 2 == 0 else variants[::-1]):
            s.set_options(v)
            prev = (i - 1) % len(batches)
            s.run(batches[prev], prev)  # same state for both variants
            t0 = time.perf_counter()
            c = s.run(b, i)
            times[v].append(time.perf_counter() - t0)
        edges.append(c["num_edges"])
    ratio = np.array(times[DEFAULT_OPTIONS]) / np.array(times[0])
    res = dict(load_s=load_s, v1_median_ms=float(np.median(times[0]) * 1e3),
               final_median_ms=float(np.median(times[DEFAULT_OPTIONS]) * 1e3),
               paired_ratio_final_vs_v1=float(np.median(ratio)), mean_edges=float(np.mean(edges)),
               final_options=DEFAULT_OPTIONS)
    del s
    gc.collect()

    # GPU stage times (profiling queue; the first sampler is freed so only one device copy
    # of the graph exists at a time).
    p = _C.Sampler.from_files(*paths, FANOUTS, 1024, "gpu", True)
    p.set_options(DEFAULT_OPTIONS)
    p.run(batches[0], 0)
    p.set_profile(2)
    stages = {}
    for i, b in enumerate(batches[:20]):
        p.run(b, i)
        for k, v in p.stage_times():
            stages.setdefault(k, []).append(v)
    res["stages_ms"] = {k: float(np.median(v) * 1e3) for k, v in stages.items()}
    return res


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--graph", default="twitter")
    a.add_argument("--mode", choices=["pyg", "ours"], required=True)
    a.add_argument("--batches", type=int, default=40)
    a.add_argument("--threads", type=int, default=7)
    a.add_argument("--json-out", default=None, help="append the result here instead")
    args = a.parse_args()
    g = load(args.graph, extras=False, mmap=True)
    batches = seed_batches(g, args.batches, 1024)
    print(f"{args.graph}: {g.num_nodes:,} nodes, {len(g.indices):,} edges, "
          f"{len(batches)} batches", flush=True)
    res = run_pyg(g, batches, args.threads) if args.mode == "pyg" else run_ours(g, batches)
    res.update(graph=args.graph, mode=args.mode, batches=len(batches), fanouts=FANOUTS)
    print(json.dumps(res, indent=2), flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(args.json_out or RESULTS / "scale.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
