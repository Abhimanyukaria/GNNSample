"""A/B of D22: today's sampler (one minibatch per call) vs K minibatches per launch
("batched") vs one minibatch per work-group ("megakernel").

The same M minibatches (batch 1024, [15,10,5], the graph's train split or a random 1%) go
through every configuration; rounds alternate the configuration order to spread drift.
Reported per configuration: minibatches/s (throughput) and the wall time of one call, which
is the latency until its minibatches are available (all K arrive together).
Appends one JSON line per configuration to --json-out (default results/multi_ab.jsonl).

Usage: python bench/multi_ab.py --graph products [--device gpu] [--batched 1,2,4,8]
                                 [--mega 8,16,32] [--wg 512] [--minibatches 64] [--rounds 2]
                                 [--sched 4,8,16] [--hub-mb 4]
D23: --sched adds "batched+sched" (cross-minibatch locality schedule) at those K; --graph
may be a reorder.py numbering ("products@degree"), and --hub-mb > 0 turns on the hub cache
hints there (every configuration, today's included).
CPU: --device cpu, with CL_CONFIG_CPU_TBB_NUM_WORKERS / DPCPP_CPU_NUM_CUS set by --threads.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default="products")
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--threads", type=int, default=0, help="CPU runtime threads (0 = default)")
ap.add_argument("--batched", default="1,2,4,8")
ap.add_argument("--mega", default="8,16,32")
ap.add_argument("--wg", type=int, default=512)
ap.add_argument("--minibatches", type=int, default=64)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--sched", default="", help="K values for batched + locality schedule")
ap.add_argument("--hub-mb", type=float, default=0.0, help="hub cache hints (hub-first graphs)")
ap.add_argument("--tables", default="", help="D25: K values for batched with direct and with "
                "hash dedup tables (configs batched@direct, batched@hash)")
ap.add_argument("--json-out", default=None)
args = ap.parse_args()
if args.threads:
    os.environ["CL_CONFIG_CPU_TBB_NUM_WORKERS"] = str(args.threads)
    os.environ["DPCPP_CPU_NUM_CUS"] = str(args.threads)

import numpy as np  # noqa: E402

from gnnsample import _C  # noqa: E402
from gnnsample.graph import load  # noqa: E402
from gnnsample.loader import DEFAULT_OPTIONS  # noqa: E402
from gnnsample.reorder import hub_count  # noqa: E402

FANOUTS = [15, 10, 5]
ROOT = Path(__file__).resolve().parents[1]
g = load(args.graph, extras=False)
tr = ROOT / f"data/csc/{g.meta.get('base', args.graph)}/train_idx.npy"
HUBS = hub_count(g.indptr, int(args.hub_mb * (1 << 20))) if args.hub_mb > 0 else 0
rng = np.random.default_rng(0)
pool = np.load(tr) if tr.exists() else rng.choice(g.num_nodes, int(0.01 * g.num_nodes), replace=False)
pool = rng.permutation(pool)
M = min(args.minibatches, len(pool) // 1024)
mbs = [pool[i * 1024:(i + 1) * 1024].astype(np.int32) for i in range(M)]
keys = [1000 + i for i in range(M)]

configs = [("today", 1)] + [("batched", int(k)) for k in args.batched.split(",") if k] + \
          [("batched+sched", int(k)) for k in args.sched.split(",") if k] + \
          [(f"batched@{t}", int(k)) for k in args.tables.split(",") if k
           for t in ("direct", "hash")] + \
          [("megakernel", int(k)) for k in args.mega.split(",") if k]
results = {c: dict(calls=[], edges=0) for c in configs}


def run_config(mode, K):
    if mode == "today":
        s = _C.Sampler(g.indptr, g.indices, FANOUTS, 1024, args.device)
        s.set_options(DEFAULT_OPTIONS | (_C.HUB_HINTS if HUBS else 0))
        s.set_hubs(HUBS)
        if g.order is not None:
            s.set_reorder(g.order)
        s.run(mbs[0], 0)  # warm-up
        calls, edges = [], 0
        for i in range(M):
            t0 = time.perf_counter()
            c = s.run(mbs[i], keys[i])
            calls.append(time.perf_counter() - t0)
            edges += c["num_edges"]
        return calls, edges
    s = _C.MultiSampler(g.indptr, g.indices, FANOUTS, 1024, K, args.device)
    if g.order is not None:
        s.set_reorder(g.order)
    s.set_hubs(HUBS, HUBS > 0)
    s.set_schedule(mode == "batched+sched")
    if "@" in mode:  # D25: dedup table kind
        mode, table = mode.split("@")
        s.set_table(table)
    mode = "batched" if mode == "batched+sched" else mode
    s.run(mbs[:K], keys[:K], mode, args.wg)  # warm-up (JIT, first table reset)
    calls, edges = [], 0
    for i in range(0, M - K + 1, K):
        t0 = time.perf_counter()
        counts = s.run(mbs[i:i + K], keys[i:i + K], mode, args.wg)
        calls.append(time.perf_counter() - t0)
        edges += sum(sum(ne) for _, ne in counts)
    return calls, edges


for r in range(args.rounds):
    order = configs if r % 2 == 0 else configs[::-1]
    for mode, K in order:
        try:
            calls, edges = run_config(mode, K)
        except Exception as e:  # noqa: BLE001 - e.g. allocation failure at large K
            results[(mode, K)]["error"] = f"{type(e).__name__}: {e}"
            continue
        results[(mode, K)]["calls"] += calls
        results[(mode, K)]["edges"] += edges
        gc.collect()

print(f"{args.graph} on {args.device}{' T=' + str(args.threads) if args.threads else ''}: "
      f"{M} minibatches x {args.rounds} rounds")
print(f"{'config':16} {'minibatches/s':>14} {'vs today':>9} {'call latency ms':>16} {'M edges/s':>10}")
base = None
out = []
for (mode, K), v in results.items():
    if "error" in v and not v["calls"]:
        print(f"{mode + ' K=' + str(K):16} error: {v['error'][:70]}")
        out.append(dict(mode=mode, K=K, error=v["error"]))
        continue
    total = sum(v["calls"])
    mb = len(v["calls"]) * K
    tput = mb / total
    base = base or tput
    lat = float(np.median(v["calls"]) * 1e3)
    print(f"{mode + ' K=' + str(K):16} {tput:14.1f} {tput / base:8.2f}x {lat:16.1f} "
          f"{v['edges'] / total / 1e6:10.1f}")
    out.append(dict(mode=mode, K=K, minibatches_per_s=tput, speedup_vs_today=tput / base,
                    call_latency_ms=lat, medges_per_s=v["edges"] / total / 1e6))
dest = Path(args.json_out) if args.json_out else ROOT / "results/multi_ab.jsonl"
with open(dest, "a") as f:
    f.write(json.dumps(dict(graph=args.graph, device=args.device, threads=args.threads, hubs=HUBS,
                            wg=args.wg, minibatches=M, rounds=args.rounds, rows=out)) + "\n")
