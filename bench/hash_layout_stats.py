"""What would direct-mapped dedup (node id = table index) save over the hash table?

For real minibatches (batch 1024, [15,10,5]), replaying the inserts in edge order on the
host with the GPU's hash function (murmur3 finalizer, linear probing, small-table mask):
  probes/insert   slots inspected per insert (1 = first slot)
  claims          inserts that claim a new slot (each one is a 64-bit CAS on the GPU)
  lines hashed    distinct 64-byte lines touched by the hash table (16-byte slots)
  lines direct    distinct 64-byte lines a direct-mapped array would touch (8-byte slots)
  direct MB       size of the direct-mapped array (8 bytes per graph node)

Usage: python bench/hash_layout_stats.py [--graphs products,livejournal,orkut] [--batches 5]
"""

import argparse

import numpy as np

from gnnsample import _C
from gnnsample.graph import load

FANOUTS = [15, 10, 5]


def mix(x):
    x = x.astype(np.uint64) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(13)
    x = (x * np.uint64(0xC2B2AE35)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    return x.astype(np.int64)


def replay(keys, cap):
    """Linear probing in insertion order; returns (total probes, claims, slots used)."""
    table = np.full(cap, -1, dtype=np.int64)
    mask = cap - 1
    probes = claims = 0
    slots = []
    for k, h in zip(keys.tolist(), (mix(keys) & mask).tolist()):
        s = h
        while True:
            probes += 1
            cur = table[s]
            if cur == k:
                break
            if cur == -1:
                table[s] = k
                claims += 1
                slots.append(s)
                break
            s = (s + 1) & mask
    return probes, claims, np.array(slots)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", default="products,livejournal,orkut")
    p.add_argument("--batches", type=int, default=5)
    args = p.parse_args()
    for name in args.graphs.split(","):
        g = load(name, extras=False)
        s = _C.Sampler(g.indptr, g.indices, FANOUTS, 1024, "gpu")
        cap = s.max_nodes  # small table: cap/2 of the full table, = next pow2 >= max_nodes
        c = 1
        while c < s.max_nodes:
            c <<= 1
        cap = c
        seeds = np.random.default_rng(0).permutation(np.asarray(g.train_seeds()))
        tot = dict(inserts=0, probes=0, claims=0, lines_hash=0, lines_direct=0, nodes=0)
        for b in range(args.batches):
            r = s.sample(seeds[b * 1024:(b + 1) * 1024].astype(np.int32), b)
            keys = np.concatenate([r["node"][: len(r["num_sampled_nodes"]) and 1024],
                                   r["node"][r["row"]]])  # seeds, then every edge's key
            probes, claims, slots = replay(keys, cap)
            tot["inserts"] += len(keys)
            tot["probes"] += probes
            tot["claims"] += claims
            tot["lines_hash"] += len(np.unique(slots // 4))           # 16 B slots
            tot["lines_direct"] += len(np.unique(r["node"] // 8))     # 8 B slots
            tot["nodes"] += len(r["node"])
        n = args.batches
        print(f"{name}: table {cap:,} slots ({cap * 16 / 1e6:.0f} MB) vs direct "
              f"{g.num_nodes:,} x 8 B ({g.num_nodes * 8 / 1e6:.0f} MB)")
        print(f"  per batch: {tot['inserts'] / n:,.0f} inserts, probes/insert "
              f"{tot['probes'] / tot['inserts']:.3f}, claims (CAS) {tot['claims'] / n:,.0f} "
              f"({tot['claims'] / tot['inserts']:.1%} of inserts)")
        print(f"  distinct 64 B lines: hashed {tot['lines_hash'] / n:,.0f} vs direct "
              f"{tot['lines_direct'] / n:,.0f} (ratio {tot['lines_direct'] / tot['lines_hash']:.2f}) "
              f"for {tot['nodes'] / n:,.0f} nodes")


if __name__ == "__main__":
    main()
