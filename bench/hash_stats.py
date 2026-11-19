"""Edge statistics that decide which hash-phase optimizations can pay off.

Per hop, over real minibatches (batch 1024, [15,10,5]):
  old      edges whose neighbour already had a local id before this hop (seed or earlier
           hop): insert could label them directly and skip flag/relabel table reads
  winner   first occurrence of a node new in this hop (gets a fresh local id)
  dup-new  later occurrences of a node new in this hop
  sg-dup   edges whose key equals another edge's key in the same aligned group of 16
           consecutive edges (what warp-aggregated inserts could merge)
and the hash-table load (distinct nodes / slots).

Usage: python bench/hash_stats.py [--graphs products,livejournal,orkut] [--batches 10]
"""

import argparse

import numpy as np

from gnnsample import _C
from gnnsample.graph import load

FANOUTS = [15, 10, 5]
SG = 16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", default="products,livejournal,orkut")
    p.add_argument("--batches", type=int, default=10)
    args = p.parse_args()
    for name in args.graphs.split(","):
        g = load(name, extras=False)
        s = _C.Sampler(g.indptr, g.indices, FANOUTS, 1024, "gpu")
        seeds = np.random.default_rng(0).permutation(np.asarray(g.train_seeds()))
        acc = np.zeros((len(FANOUTS), 4))  # old, winner, dup-new, sg-dup
        edges = np.zeros(len(FANOUTS))
        nodes = []
        for b in range(args.batches):
            r = s.sample(seeds[b * 1024:(b + 1) * 1024].astype(np.int32), b)
            n_nodes, n_edges = r["num_sampled_nodes"], r["num_sampled_edges"]
            cum_n = np.cumsum(n_nodes)  # cum_n[h] = nodes known after hop h-1 (h=0: seeds)
            e0 = 0
            for h, ne in enumerate(n_edges):
                row = r["row"][e0:e0 + ne]
                nbr = r["node"][row]  # global ids = hash keys
                known = cum_n[h]
                old = row < known
                new_rows = row[~old]
                _, first = np.unique(new_rows, return_index=True)
                winners = len(first)
                # duplicates of a key inside each aligned group of 16 edges
                pad = (-ne) % SG
                grp = np.concatenate([nbr, np.full(pad, -1 - np.arange(pad))]).reshape(-1, SG)
                srt = np.sort(grp, axis=1)
                sg_dup = int((srt[:, 1:] == srt[:, :-1]).sum())
                acc[h] += [old.sum(), winners, (~old).sum() - winners, sg_dup]
                edges[h] += ne
                e0 += ne
            nodes.append(len(r["node"]))
        cap = 1
        while cap < 2 * min(1024 + 1024 * (15 + 150 + 750), g.num_nodes):
            cap <<= 1
        print(f"{name}: mean nodes/batch {np.mean(nodes):,.0f}, hash slots {cap:,} "
              f"(load {np.mean(nodes) / cap:.2f})")
        for h in range(len(FANOUTS)):
            f = acc[h] / edges[h]
            print(f"  hop {h}: {edges[h] / args.batches:9,.0f} edges | old {f[0]:6.1%}  "
                  f"winner {f[1]:6.1%}  dup-new {f[2]:6.1%}  sg-dup {f[3]:6.2%}")
        tot = acc.sum(0) / edges.sum()
        print(f"  all  : old {tot[0]:6.1%}  winner {tot[1]:6.1%}  dup-new {tot[2]:6.1%}  "
              f"sg-dup {tot[3]:6.2%}")


if __name__ == "__main__":
    main()
