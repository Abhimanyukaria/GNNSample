"""Hardware-independent locality counters for D23 (graph reordering, hub cache, cross-minibatch
reuse). No timing: these count which 64-byte lines the sampler's graph reads touch, so the
numbers hold for any GPU and can be projected to one we do not have.

Per graph numbering (the original and each reorder.py variant), over M training minibatches
(batch 1024, [15,10,5]), from the sampler's own output (internal ids, raw=True):
  reads           indptr reads (2 per expanded node) and neighbour reads (1 per sampled edge)
  lines           distinct 64-byte lines those reads touch within one minibatch; reads/line is
                  the spatial reuse a cache gets for free inside a minibatch
  hub share       fraction of neighbour reads that fall in the hub prefix (ids < h, h from a
                  byte budget); meaningful on hub-first numberings (degree, hubsort)
  cross-K reuse   for K minibatches sampled together: (sum of each one's distinct lines) /
                  (distinct lines of their union). 1.0 = nothing shared; K = all shared. This
                  bounds what K-per-launch (batched) can save on graph traffic.

Usage: python bench/locality_stats.py --graph products [--variants products@degree,...]
       [--minibatches 32] [--device gpu|cpu]
Appends one JSON line to --json-out (default results/locality_stats.jsonl).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gnnsample import _C
from gnnsample.graph import load
from gnnsample.reorder import hub_count

FANOUTS = [15, 10, 5]
ROOT = Path(__file__).resolve().parents[1]
LINE = 64
BUDGETS_MB = (1, 4, 16)
KS = (2, 4, 8, 16)


def minibatches(g, count, batch=1024):
    base = g.meta.get("base", g.name)
    tr = ROOT / f"data/csc/{base}/train_idx.npy"
    pool = np.load(tr) if tr.exists() else g.train_seeds()
    pool = np.random.default_rng(0).permutation(pool)
    n = min(count, len(pool) // batch)
    return [pool[i * batch:(i + 1) * batch].astype(np.int32) for i in range(n)]


def lines_of(g, r):
    """Distinct indptr and indices lines one minibatch touches, plus its read counts and
    the target ids of its neighbour reads."""
    node, col, edge = r["node"], r["col"], r["edge"]
    n_expanded = len(node) - r["num_sampled_nodes"][-1]  # the last hop's new nodes are leaves
    v = node[:n_expanded].astype(np.int64)
    ip = np.unique(np.concatenate([(v * 8) // LINE, ((v + 1) * 8) // LINE]))
    ix = np.unique((edge * 4) // LINE)
    return ip, ix, 2 * n_expanded, len(edge), node[col]


def stats(name, args):
    g = load(name, extras=False, mmap=True)
    s = _C.Sampler(np.asarray(g.indptr), np.asarray(g.indices), FANOUTS, 1024, args.device)
    s.set_options(8169)
    if g.order is not None:
        s.set_reorder(g.order)
    mbs = minibatches(g, args.minibatches)
    hubs = {mb: hub_count(g.indptr, mb << 20) for mb in BUDGETS_MB}
    per = []
    for i, seeds in enumerate(mbs):
        per.append(lines_of(g, s.sample(seeds, 1000 + i, raw=True)))
    ip_reads = sum(p[2] for p in per)
    ix_reads = sum(p[3] for p in per)
    out = dict(graph=name, minibatches=len(mbs),
               indptr_reads_per_mb=ip_reads / len(per), nbr_reads_per_mb=ix_reads / len(per),
               indptr_lines_per_mb=float(np.mean([len(p[0]) for p in per])),
               nbr_lines_per_mb=float(np.mean([len(p[1]) for p in per])))
    out["indptr_reads_per_line"] = out["indptr_reads_per_mb"] / out["indptr_lines_per_mb"]
    out["nbr_reads_per_line"] = out["nbr_reads_per_mb"] / out["nbr_lines_per_mb"]
    tgt = np.concatenate([p[4] for p in per])
    out["hub_share"] = {f"{mb}MB": dict(hubs=h, nbr_read_share=float((tgt < h).mean()))
                        for mb, h in hubs.items()}
    cross = {}
    for K in KS:
        if K > len(per):
            break
        r_ip, r_ix = [], []
        for j in range(0, len(per) - K + 1, K):
            grp = per[j:j + K]
            r_ip.append(sum(len(p[0]) for p in grp) / len(np.unique(np.concatenate([p[0] for p in grp]))))
            r_ix.append(sum(len(p[1]) for p in grp) / len(np.unique(np.concatenate([p[1] for p in grp]))))
        cross[K] = dict(indptr=float(np.mean(r_ip)), nbr=float(np.mean(r_ix)))
    out["cross_k_reuse"] = cross
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="products")
    ap.add_argument("--variants", default="", help="extra numberings, e.g. products@degree")
    ap.add_argument("--minibatches", type=int, default=32)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    names = [args.graph] + [v for v in args.variants.split(",") if v]
    rows = []
    for name in names:
        r = stats(name, args)
        rows.append(r)
        hs = "  ".join(f"{k}:{v['nbr_read_share']:.2f}" for k, v in r["hub_share"].items())
        cx = "  ".join(f"K={k}:{v['nbr']:.2f}/{v['indptr']:.2f}" for k, v in r["cross_k_reuse"].items())
        print(f"{name:20} indptr {r['indptr_lines_per_mb']:9.0f} lines ({r['indptr_reads_per_line']:.2f} rd/ln)"
              f"  nbr {r['nbr_lines_per_mb']:9.0f} lines ({r['nbr_reads_per_line']:.2f} rd/ln)")
        print(f"{'':20} hub share {hs}")
        print(f"{'':20} cross-K reuse nbr/indptr {cx}")
    dest = Path(args.json_out) if args.json_out else ROOT / "results/locality_stats.jsonl"
    with open(dest, "a") as f:
        f.write(json.dumps(dict(rows=rows)) + "\n")


if __name__ == "__main__":
    main()
