"""Synthetic power-law graphs for tests and quick benchmarks (no download needed)."""

import numpy as np


def power_law_csc(num_nodes: int, avg_degree: float, alpha: float = 2.1, seed: int = 0):
    """Chung-Lu style directed graph with power-law in-degrees, as CSC (indptr, indices).
    Duplicate edges and self loops are removed, as in data/prepare.py."""
    rng = np.random.default_rng(seed)
    w = (np.arange(1, num_nodes + 1, dtype=np.float64)) ** (-1.0 / (alpha - 1.0))
    w = w / w.sum()
    m = int(num_nodes * avg_degree)
    dst = rng.choice(num_nodes, m, p=w)
    src = rng.integers(0, num_nodes, m)
    keep = src != dst
    key = np.unique(dst[keep].astype(np.int64) * num_nodes + src[keep])
    dst, src = np.divmod(key, num_nodes)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(dst, minlength=num_nodes), out=indptr[1:])
    return indptr, src.astype(np.int32)
