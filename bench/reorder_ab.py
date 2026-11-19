"""Paired A/B of D23 on today's sampler: the original numbering vs reorder.py's numberings,
with and without the hub cache hints (HUB_HINTS), all in one process.

Every variant samples the same minibatches with the same keys (bit-identical output, see
tests/test_reorder.py), in rotating order, so the iGPU's drift hits all variants alike
(rule 3 in CLAUDE.md). Reported per variant: median per-batch time and the median of
t_variant / t_original over minibatches.

Each variant holds its own device copy of the graph (products ~0.5 GB, Orkut ~0.94 GB), so
pick --methods to fit memory; run the reorderings first:
  python -m gnnsample.reorder --graph products --method degree   (and hubsort, rcm, random)

Usage: python bench/reorder_ab.py --graph products --methods degree,hubsort,rcm,random
       [--hub-mb 1,4,16] [--batches 40]
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np

from gnnsample import _C
from gnnsample.graph import load
from gnnsample.loader import DEFAULT_OPTIONS
from gnnsample.reorder import hub_count

FANOUTS = [15, 10, 5]
ROOT = Path(__file__).resolve().parents[1]
HUB_FIRST = ("degree", "hubsort")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="products")
    ap.add_argument("--methods", default="degree,hubsort")
    ap.add_argument("--hub-mb", default="4", help="hub budgets (MB) for HUB_HINTS variants")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tr = ROOT / f"data/csc/{args.graph}/train_idx.npy"
    variants = []  # (label, sampler, options)
    g = load(args.graph, extras=False, mmap=True)
    pool = np.load(tr) if tr.exists() else g.train_seeds()
    pool = np.random.default_rng(0).permutation(pool)
    n = min(args.batches, len(pool) // 1024)
    batches = [pool[i * 1024:(i + 1) * 1024].astype(np.int32) for i in range(n)]

    def add(name):
        gg = load(name, extras=False, mmap=True)
        s = _C.Sampler(np.asarray(gg.indptr), np.asarray(gg.indices), FANOUTS, 1024, "gpu")
        if gg.order is not None:
            s.set_reorder(gg.order)
        variants.append((name, s, DEFAULT_OPTIONS, 0))
        method = gg.meta.get("reorder")
        if method in HUB_FIRST:
            for mb in (float(x) for x in args.hub_mb.split(",") if x):
                variants.append((f"{name}+hubs{mb:g}MB", s, DEFAULT_OPTIONS | _C.HUB_HINTS,
                                 hub_count(gg.indptr, int(mb * (1 << 20)))))
        del gg
        gc.collect()

    add(args.graph)
    for m in (x for x in args.methods.split(",") if x):
        add(f"{args.graph}@{m}")

    for _, s, opts, hubs in variants:  # warm-up: JIT, first table reset
        s.set_options(opts)
        s.set_hubs(hubs)
        s.run(batches[0], 0)
    times = {v[0]: [] for v in variants}
    for i, b in enumerate(batches):
        order = variants[i % len(variants):] + variants[:i % len(variants)]
        for label, s, opts, hubs in order:
            s.set_options(opts)
            s.set_hubs(hubs)
            t0 = time.perf_counter()
            s.run(b, i)
            times[label].append(time.perf_counter() - t0)

    base = np.array(times[args.graph])
    print(f"{args.graph}: {len(batches)} minibatches, today's sampler (options {DEFAULT_OPTIONS})")
    print(f"{'variant':34} {'median ms':>10} {'vs original':>12} {'IQR':>16}")
    rows = []
    for label, _, opts, hubs in variants:
        t = np.array(times[label])
        r = t / base
        rows.append(dict(variant=label, options=opts, hubs=hubs, median_ms=float(np.median(t) * 1e3),
                         paired_ratio=float(np.median(r)),
                         ratio_iqr=[float(np.percentile(r, 25)), float(np.percentile(r, 75))]))
        print(f"{label:34} {np.median(t) * 1e3:10.2f} {np.median(r):12.3f} "
              f"{np.percentile(r, 25):7.3f}-{np.percentile(r, 75):.3f}")
    dest = Path(args.json_out) if args.json_out else ROOT / "results/reorder_ab.jsonl"
    with open(dest, "a") as f:
        f.write(json.dumps(dict(graph=args.graph, batches=len(batches), rows=rows)) + "\n")


if __name__ == "__main__":
    main()
