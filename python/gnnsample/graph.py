"""Load a prepared CSC graph (data/prepare.py output) and its optional OGB extras."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "csc"


@dataclass
class Graph:
    name: str
    indptr: np.ndarray
    indices: np.ndarray
    meta: dict
    extras: dict = field(default_factory=dict)  # feat, label, train_idx, ... (original ids)
    order: np.ndarray = None  # reordered graphs ("name@method"): order[new id] = original id

    @property
    def num_nodes(self) -> int:
        return len(self.indptr) - 1

    def reorder_args(self, with_pos_shift=True):
        """(order, pos_shift) for Sampler.set_reorder; (None, None) for an original graph.
        pos_shift loads the original indptr (8 bytes per node)."""
        if self.order is None:
            return None, None
        if not with_pos_shift:
            return self.order, None
        from .reorder import pos_shift

        base = np.memmap(DATA / self.meta["base"] / "indptr.int64.bin", dtype=np.int64, mode="r")
        return self.order, pos_shift(base, np.asarray(self.indptr), self.order)

    def train_seeds(self, frac: float = 0.01, seed: int = 0) -> np.ndarray:
        """OGB train split if present, else a random `frac` of nodes (gSampler convention)."""
        if "train_idx" in self.extras:
            return self.extras["train_idx"]
        rng = np.random.default_rng(seed)
        return rng.choice(self.num_nodes, int(frac * self.num_nodes), replace=False)


def load(name: str, extras: bool = True, mmap: bool = False) -> Graph:
    """name may be "graph@method", reorder.py's renumbering of "graph": its extras (features,
    labels, splits) are the original graph's, in original ids, and Graph.order holds the
    mapping the sampler needs."""
    d = DATA / name
    mode = "r" if mmap else None
    if mmap:
        indptr = np.memmap(d / "indptr.int64.bin", dtype=np.int64, mode=mode)
        indices = np.memmap(d / "indices.int32.bin", dtype=np.int32, mode=mode)
    else:
        indptr = np.fromfile(d / "indptr.int64.bin", dtype=np.int64)
        indices = np.fromfile(d / "indices.int32.bin", dtype=np.int32)
    g = Graph(name, indptr, indices, json.loads((d / "meta.json").read_text()))
    if "base" in g.meta:
        g.order = np.fromfile(d / "order.int32.bin", dtype=np.int32)
        d = DATA / g.meta["base"]
    if extras:
        for f in d.glob("*.npy"):
            g.extras[f.stem] = np.load(f, mmap_mode=mode)
            if not g.extras[f.stem].flags.c_contiguous:
                raise ValueError(f"{f} is not row-major; re-run data/prepare.py {name}")
    return g
