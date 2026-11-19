"""D25: where does the batched sampler's time go? Per-kernel GPU profile, batched vs single.

batched  MultiSampler batched mode, every command timed on a profiling queue (no barriers
         added): per stage (summed over hops) busy = kernel execution time; idle = time
         the GPU sat between commands (launch + drain gaps); host = the run() call's wall
         time minus the GPU span (Python, argument copies, submission, final wait).
         Reported per minibatch (block totals / K) for K = 1 and the given K.
single   today's Sampler, profile mode 2 (GPU timestamps of barrier markers): per-stage
         time including its gaps, and the run() wall time.
Same minibatches and keys for both (train split, or a random 1% for graphs without one).
Each sampler runs in its own phase (one device copy of the graph at a time).

Usage: python bench/multi_profile.py --graph products [--K 8] [--blocks 8]
Appends one JSON line to --json-out (default results/multi_profile.jsonl).
"""

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from gnnsample import _C
from gnnsample.graph import load
from gnnsample.loader import DEFAULT_OPTIONS

FANOUTS = [15, 10, 5]
ROOT = Path(__file__).resolve().parents[1]


def profile_batched(g, mbs, K, blocks):
    ms = _C.MultiSampler(g.indptr, g.indices, FANOUTS, 1024, K, "gpu", True)
    keys = list(range(len(mbs)))
    ms.run(mbs[:K], keys[:K], "batched")  # warm-up: JIT, first table reset
    ms.run(mbs[K:2 * K], keys[K:2 * K], "batched")
    ms.set_profile(True)
    busy = defaultdict(float)  # stage -> ns, summed over blocks
    per_hop = defaultdict(float)  # (hop, stage) -> ns
    n_cmd = defaultdict(int)
    span = idle = host = 0.0
    used = 0
    for b in range(blocks):
        lo = (2 + b) * K
        if lo + K > len(mbs):
            break
        t0 = time.perf_counter()
        ms.run(mbs[lo:lo + K], keys[lo:lo + K], "batched")
        wall = (time.perf_counter() - t0) * 1e9
        tr = sorted(ms.trace(), key=lambda r: r[2])
        prev_end = tr[0][2]
        for name, hop, start, end in tr:
            busy[name] += end - start
            per_hop[(hop, name)] += end - start
            n_cmd[name] += 1
            idle += max(0.0, start - prev_end)
            prev_end = max(prev_end, end)
        s = tr[-1][3] - tr[0][2]
        span += s
        host += max(0.0, wall - s)
        used += 1
    ms.set_profile(False)
    del ms
    gc.collect()
    n_mb = used * K
    ms_ = lambda ns: ns / n_mb / 1e6  # noqa: E731 - ns per block total -> ms per minibatch
    return dict(K=K, blocks=used,
                stage_ms={k: ms_(v) for k, v in sorted(busy.items(), key=lambda kv: -kv[1])},
                commands_per_minibatch={k: v / n_mb for k, v in n_cmd.items()},
                hop_stage_ms={f"{h}:{k}": ms_(v) for (h, k), v in sorted(per_hop.items())},
                busy_ms=ms_(sum(busy.values())), idle_ms=ms_(idle), gpu_span_ms=ms_(span),
                host_ms=ms_(host), wall_ms=ms_(span + host))


def profile_single(g, mbs, n):
    s = _C.Sampler(g.indptr, g.indices, FANOUTS, 1024, "gpu", True)
    s.set_options(DEFAULT_OPTIONS)
    for i in range(2):
        s.run(mbs[i], i)
    stage = defaultdict(float)
    wall = 0.0
    s.set_profile(2)
    for i in range(2, 2 + n):
        t0 = time.perf_counter()
        s.run(mbs[i], i)
        wall += time.perf_counter() - t0
        for k, v in s.stage_times():
            stage[k] += v
    s.set_profile(0)
    del s
    gc.collect()
    st = {k: v / n * 1e3 for k, v in sorted(stage.items(), key=lambda kv: -kv[1])}
    return dict(stage_ms=st, gpu_ms=sum(st.values()), wall_ms=wall / n * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="products")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    g = load(args.graph, extras=False)
    tr = ROOT / f"data/csc/{g.meta.get('base', args.graph)}/train_idx.npy"
    rng = np.random.default_rng(0)
    # No train split (LiveJournal, Orkut): random seeds, enough for every block (profiling
    # needs more than the 1% convention's 30-47 minibatches).
    pool = (np.load(tr) if tr.exists()
            else rng.choice(g.num_nodes, (args.blocks + 2) * args.K * 1024, replace=False))
    pool = rng.permutation(pool)
    n = len(pool) // 1024
    mbs = [pool[i * 1024:(i + 1) * 1024].astype(np.int32) for i in range(n)]
    blocks = min(args.blocks, n // args.K - 2)

    out = dict(graph=args.graph, minibatches=n)
    out["single"] = profile_single(g, mbs, blocks * args.K)
    out["batched_K1"] = profile_batched(g, mbs, 1, blocks * args.K)
    out[f"batched_K{args.K}"] = profile_batched(g, mbs, args.K, blocks)

    print(f"{args.graph}: ms per minibatch (batch 1024, [15,10,5]); stages summed over hops")
    sgl = out["single"]
    print(f"  single: wall {sgl['wall_ms']:.2f}  GPU stages {sgl['gpu_ms']:.2f}")
    print("    " + "  ".join(f"{k} {v:.2f}" for k, v in sgl["stage_ms"].items()))
    for key in ("batched_K1", f"batched_K{args.K}"):
        r = out[key]
        print(f"  {key}: wall {r['wall_ms']:.2f} = GPU busy {r['busy_ms']:.2f} + GPU idle "
              f"{r['idle_ms']:.2f} + host {r['host_ms']:.2f}")
        print("    " + "  ".join(f"{k} {v:.2f}" for k, v in r["stage_ms"].items()))
    dest = Path(args.json_out) if args.json_out else ROOT / "results/multi_profile.jsonl"
    with open(dest, "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
