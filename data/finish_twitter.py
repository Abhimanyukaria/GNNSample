"""Validate the CSC written by tools/convert_twitter.sh and write its meta.json.

Checks: indptr has n+1 non-decreasing entries starting at 0, ends at the indices length,
every neighbour id is in range, and a sample of rows is sorted and free of self loops.
Usage: python data/finish_twitter.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "twitter" / "twitter-2010-t.graph"
OUT = ROOT / "csc" / "twitter"
N_NODES = 41_652_230  # from the LAW dataset page / converter log


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main():
    indptr = np.memmap(OUT / "indptr.int64.bin", dtype=np.int64, mode="r")
    indices = np.memmap(OUT / "indices.int32.bin", dtype=np.int32, mode="r")
    n = len(indptr) - 1
    assert n == N_NODES, f"expected {N_NODES} nodes, got {n}"
    assert indptr[0] == 0 and indptr[-1] == len(indices), "indptr does not span indices"
    deg = np.diff(indptr)
    assert deg.min() >= 0, "indptr is not non-decreasing"
    lo, hi = np.inf, -np.inf
    for s in range(0, len(indices), 1 << 27):  # range check in 512 MB chunks
        c = indices[s:s + (1 << 27)]
        lo, hi = min(lo, int(c.min())), max(hi, int(c.max()))
    assert lo >= 0 and hi < n, f"neighbour ids out of range: {lo}..{hi}"
    rng = np.random.default_rng(0)
    for v in rng.choice(n, 2000, replace=False):
        row = indices[indptr[v]:indptr[v + 1]]
        assert np.all(np.diff(row) > 0), f"row {v} not strictly sorted"
        assert not np.any(row == v), f"row {v} has a self loop"
    meta = dict(name="twitter", num_nodes=int(n), num_edges=int(len(indices)), directed=True,
                source="http://data.law.di.unimi.it/webdata/twitter-2010/twitter-2010-t.graph "
                       "(transposed BVGraph; successors = in-neighbours)",
                sha256=sha256(RAW), max_in_degree=int(deg.max()),
                mean_in_degree=float(deg.mean()), zero_in_degree=int((deg == 0).sum()))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
