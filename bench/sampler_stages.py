"""Per-batch latency of the SYCL sampler vs pyg-lib, with a stage breakdown (Phase 3).

For each graph: the same seed batches (batch 1024, fanout [15,10,5]) through
  pyg-lib        neighbor_sample on the CPU, one call per batch (what NeighborLoader does)
  sycl run       our sampler, results left on the device
  sycl pipeline  run + export to torch XPU tensors + feature gather (D11), when the graph has
                 features (products) or with random 128-d features otherwise
and a profiled pass (waits between stages) giving where the sycl run time goes.
Results append to results/sampler_stages.jsonl tagged with --tag, so every optimization
is one comparable line.

Usage: python bench/sampler_stages.py --tag v1 [--graphs products,livejournal,orkut]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pyg_lib.sampler import neighbor_sample

from gnnsample.graph import load
from gnnsample.loader import GpuSampler

FANOUTS = [15, 10, 5]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def med_ms(ts):
    return float(np.median(ts) * 1e3)


def bench_graph(name, n_batches, batch, threads, options=0):
    g = load(name)
    seeds = g.train_seeds()
    rng = np.random.default_rng(0)
    seeds = rng.permutation(np.asarray(seeds))
    n_batches = min(n_batches, len(seeds) // batch)  # full batches only
    batches = [seeds[i * batch:(i + 1) * batch] for i in range(n_batches)]
    feat = g.extras.get("feat")
    if feat is None:
        # 64-d random features: the iGPU caps one allocation at 1.07 GB (128-d LiveJournal
        # would be 2.5 GB). Width does not affect sampling; gather cost scales with it.
        feat = rng.standard_normal((g.num_nodes, 64), dtype=np.float32)

    torch.set_num_threads(threads)
    rowptr = torch.from_numpy(g.indptr)
    col = torch.from_numpy(g.indices.astype(np.int64))
    pyg, edges = [], []
    for b in batches:
        t0 = time.perf_counter()
        out = neighbor_sample(rowptr, col, torch.from_numpy(b).long(), FANOUTS, csc=True)
        pyg.append(time.perf_counter() - t0)
        edges.append(out[0].numel())
    del rowptr, col

    gs = GpuSampler(g.indptr, g.indices, FANOUTS, batch, feat=feat, usm="host", options=options)
    for i, b in enumerate(batches[:3]):  # warm-up (JIT, allocator)
        gs.sample(b, seed=i)
    run, pipe, ours_edges = [], [], []
    for i, b in enumerate(batches):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        c = gs.s.run(b.astype(np.int32), i)
        run.append(time.perf_counter() - t0)
        ours_edges.append(c["num_edges"])
        t0 = time.perf_counter()
        gs.sample(b, seed=i)
        torch.xpu.synchronize()
        pipe.append(time.perf_counter() - t0)

    # Stage breakdown from GPU timestamps (no host waits) on a separate profiling queue,
    # so the timings above are not affected by profiling.
    del gs
    gp = GpuSampler(g.indptr, g.indices, FANOUTS, batch, options=options, profiling=True)
    gp.set_profile(2)
    stages = {}
    for i, b in enumerate(batches):
        gp.s.run(b.astype(np.int32), i)
        for k, v in gp.stage_times().items():
            stages.setdefault(k, []).append(v)
    del gp

    return dict(
        graph=name, options=options, batches=n_batches, batch=batch, cpu_threads=threads,
        pyg_ms=med_ms(pyg), sycl_run_ms=med_ms(run), sycl_pipeline_ms=med_ms(pipe),
        speedup_run_vs_pyg=float(np.median(pyg) / np.median(run)),
        mean_edges_pyg=float(np.mean(edges)), mean_edges_sycl=float(np.mean(ours_edges)),
        sycl_medges_per_s=float(np.mean(ours_edges) / np.median(run) / 1e6),
        stages_ms={k: med_ms(v) for k, v in stages.items()},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--graphs", default="products,livejournal,orkut")
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--threads", type=int, default=7)
    p.add_argument("--options", type=int, default=0, help="sampler option bitmask")
    p.add_argument("--json-out", default=None, help="append results here instead")
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    for name in args.graphs.split(","):
        r = dict(tag=args.tag, **bench_graph(name, args.batches, args.batch, args.threads, args.options))
        with open(args.json_out or RESULTS / "sampler_stages.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        st = "  ".join(f"{k} {v:.1f}" for k, v in r["stages_ms"].items())
        print(f"[{args.tag}] {name:12} pyg {r['pyg_ms']:7.1f} ms | sycl run "
              f"{r['sycl_run_ms']:6.1f} ms ({r['speedup_run_vs_pyg']:.1f}x) | pipeline "
              f"{r['sycl_pipeline_ms']:6.1f} ms | {r['sycl_medges_per_s']:.0f} M edges/s\n"
              f"    GPU stage times (ms): {st}", flush=True)


if __name__ == "__main__":
    main()
