"""Paired A/B ablation of sampler options (decision D12).

The iGPU's speed drifts between runs (30.8 / 33.0 / 48.0 / 65.1 ms for the same v1
benchmark), so separate before/after runs are not comparable. Here every minibatch is
sampled once with each variant, in rotating order, in the same process. The reported
effect of a variant is the median over minibatches of t_variant / t_baseline, so drift
affects both sides of each ratio alike.

Usage: python bench/ablate.py --graph products --variants 0,1,2,4,7 [--batches 40]
  variant = bitmask of gnnsample._C.PACKED_HASH | FAST_SCAN | CLEAR_USED; the first one
  is the baseline.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pyg_lib.sampler import neighbor_sample

from gnnsample import _C
from gnnsample.graph import load
from gnnsample.reorder import hub_count

FANOUTS = [15, 10, 5]
RESULTS = Path(__file__).resolve().parents[1] / "results"
NAMES = {_C.PACKED_HASH: "packed", _C.FAST_SCAN: "fastscan", _C.CLEAR_USED: "clearused",
         _C.NO_SYNC: "nosync", _C.PHILOX_REUSE: "philox4", _C.ASSIGN_FROM_SCAN: "assignscan",
         _C.SLOT32: "slot32", _C.GEN_TAGS: "gentags",
         _C.SUBGROUP_SAMPLE: "sgsample", _C.LABEL_EARLY: "labelearly",
         _C.SMALL_TABLE: "smalltable", _C.DIRECT_MAP: "direct", _C.NO_NODE_SLOT: "nonodeslot",
         _C.HUB_HINTS: "hubhints"}


def label(mask):
    return "+".join(n for bit, n in NAMES.items() if mask & bit) or "v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="products")
    p.add_argument("--variants", default="0,1,2,4,7")
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--threads", type=int, default=7)
    p.add_argument("--json-out", default=None, help="append the result here instead")
    p.add_argument("--hub-mb", type=float, default=4.0,
                   help="HUB_HINTS: hub rows = the hub-first prefix that fits in this many MB")
    args = p.parse_args()
    variants = [int(v) for v in args.variants.split(",")]

    g = load(args.graph, extras=False)  # "graph@method": reorder.py's renumbering (D23)
    base = g.meta.get("base", args.graph)
    tr = Path(__file__).resolve().parents[1] / f"data/csc/{base}/train_idx.npy"
    seeds = np.load(tr) if tr.exists() else g.train_seeds()
    seeds = np.random.default_rng(0).permutation(seeds)
    n_full = len(seeds) // args.batch  # only full batches (Orkut's 1% seeds give 30)
    if n_full < args.batches:
        print(f"only {n_full} full batches available; using {n_full}")
    batches = [seeds[i * args.batch:(i + 1) * args.batch].astype(np.int32)
               for i in range(min(args.batches, n_full))]

    torch.set_num_threads(args.threads)
    rowptr, col = torch.from_numpy(g.indptr), torch.from_numpy(g.indices.astype(np.int64))
    pyg = []
    for b in batches:
        t0 = time.perf_counter()
        neighbor_sample(rowptr, col, torch.from_numpy(b).long(), FANOUTS, csc=True)
        pyg.append(time.perf_counter() - t0)
    del rowptr, col

    s = _C.Sampler(g.indptr, g.indices, FANOUTS, args.batch, "gpu")
    if g.order is not None:
        s.set_reorder(g.order)
    s.set_hubs(hub_count(g.indptr, int(args.hub_mb * (1 << 20))))
    for v in variants:  # warm-up: JIT and first full table reset for each variant
        s.set_options(v)
        s.run(batches[0], 0)
        s.run(batches[1], 1)
    times = {v: [] for v in variants}
    for i, b in enumerate(batches):
        order = variants[i % len(variants):] + variants[: i % len(variants)]
        for v in order:
            s.set_options(v)
            # CLEAR_USED must clean up after a batch sampled with the same layout; run the
            # previous batch first (untimed) so every variant starts from its own state.
            prev = (i - 1) % len(batches)
            s.run(batches[prev], prev)
            t0 = time.perf_counter()
            s.run(b, i)
            times[v].append(time.perf_counter() - t0)

    base = np.array(times[variants[0]])
    pyg_med = float(np.median(pyg))
    print(f"{args.graph}: {args.batches} batches; pyg-lib median {pyg_med * 1e3:.1f} ms")
    print(f"{'variant':28} {'median ms':>10} {'vs ' + label(variants[0]):>12} {'vs pyg':>8}")
    rows = []
    for v in variants:
        t = np.array(times[v])
        ratio = float(np.median(t / base))
        rows.append(dict(options=v, label=label(v), median_ms=float(np.median(t) * 1e3),
                         paired_ratio_vs_base=ratio,
                         ratio_iqr=[float(np.percentile(t / base, 25)),
                                    float(np.percentile(t / base, 75))],
                         speedup_vs_pyg=float(np.median(np.array(pyg) / t))))
        print(f"{label(v):28} {np.median(t) * 1e3:10.1f} {ratio:12.3f} "
              f"{rows[-1]['speedup_vs_pyg']:7.2f}x")
    RESULTS.mkdir(exist_ok=True)
    with open(args.json_out or RESULTS / "ablation.jsonl", "a") as f:
        f.write(json.dumps(dict(graph=args.graph, batches=args.batches, batch=args.batch,
                                pyg_median_ms=pyg_med * 1e3, variants=rows)) + "\n")


if __name__ == "__main__":
    main()
