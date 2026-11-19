"""CPU sampling at equal core counts: our SYCL sampler on the CPU device vs pyg-lib.

Run once per (sampler, thread count) in its own process, because the SYCL CPU runtime reads
its thread count from the environment at start-up:
  --sampler ours --threads T   our sampler, one minibatch at a time, with the runtime
                               limited to T worker threads (CL_CONFIG_CPU_TBB_NUM_WORKERS and
                               DPCPP_CPU_NUM_CUS, set here before the runtime loads)
  --sampler pyg --threads T    pyg-lib the way NeighborLoader parallelises it: T DataLoader
                               worker processes, one minibatch per call, one thread each
                               (T = 0: one call at a time in the main process)
Same seed batches for both (train split if the graph has one, else a random 1% of nodes),
batch 1024, fanout [15,10,5], sampling only (no feature gather). Reports batches/s and,
for ours, the per-batch latency. Appends to results/cpu_scaling.jsonl (or --json-out).

Usage: python bench/cpu_scaling.py --graph products --sampler ours --threads 4
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default="products")
ap.add_argument("--sampler", choices=["ours", "pyg"], required=True)
ap.add_argument("--threads", type=int, required=True)
ap.add_argument("--batches", type=int, default=48)
ap.add_argument("--json-out", default=None)
args = ap.parse_args()
if args.sampler == "ours":  # must happen before the SYCL runtime is loaded
    os.environ["CL_CONFIG_CPU_TBB_NUM_WORKERS"] = str(args.threads)
    os.environ["DPCPP_CPU_NUM_CUS"] = str(args.threads)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gnnsample.graph import load  # noqa: E402

FANOUTS = [15, 10, 5]
ROOT = Path(__file__).resolve().parents[1]

g = load(args.graph, extras=False)
tr = ROOT / f"data/csc/{args.graph}/train_idx.npy"
rng = np.random.default_rng(0)
seeds = np.load(tr) if tr.exists() else rng.choice(g.num_nodes, int(0.01 * g.num_nodes),
                                                   replace=False)
seeds = rng.permutation(seeds)
n = min(args.batches, len(seeds) // 1024)
batches = [seeds[i * 1024:(i + 1) * 1024].astype(np.int32) for i in range(n)]
warm = 4

if args.sampler == "ours":
    from gnnsample import _C
    from gnnsample.loader import DEFAULT_OPTIONS

    s = _C.Sampler(g.indptr, g.indices, FANOUTS, 1024, "cpu")
    s.set_options(DEFAULT_OPTIONS)
    for i in range(warm):
        s.run(batches[i], i)
    lat, edges = [], 0
    t0 = time.perf_counter()
    for i, b in enumerate(batches):
        t1 = time.perf_counter()
        edges += s.run(b, i)["num_edges"]
        lat.append(time.perf_counter() - t1)
    wall = time.perf_counter() - t0
    res = dict(latency_ms=float(np.median(lat) * 1e3))
else:
    from pyg_lib.sampler import neighbor_sample

    torch.set_num_threads(1)
    rowptr = torch.from_numpy(g.indptr)
    col = torch.from_numpy(g.indices.astype(np.int64))

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(batches)

        def __getitem__(self, i):
            torch.set_num_threads(1)
            out = neighbor_sample(rowptr, col, torch.from_numpy(batches[i]).long(), FANOUTS,
                                  csc=True)
            return out[0].numel()

    def epoch():
        loader = torch.utils.data.DataLoader(DS(), batch_size=None, num_workers=args.threads,
                                             persistent_workers=False)
        it = iter(loader)  # workers forked and warmed up before timing
        first = [next(it) for _ in range(min(warm, len(batches)))]
        t0 = time.perf_counter()
        rest = list(it)
        return time.perf_counter() - t0, sum(first) + sum(rest), len(rest)

    wall, edges, timed = epoch()
    n = timed
    res = {}

res.update(graph=args.graph, sampler=args.sampler, threads=args.threads, batches=n,
           batches_per_s=float(n / wall), medges_per_s=float(edges / wall / 1e6)
           if args.sampler == "ours" else None, host=platform.processor() or "x86_64")
line = json.dumps(res)
print(line, flush=True)
out = Path(args.json_out) if args.json_out else ROOT / "results/cpu_scaling.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "a") as f:
    f.write(line + "\n")
