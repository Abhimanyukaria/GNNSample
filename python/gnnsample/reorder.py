"""Graph reordering (decision D23): renumber nodes so that the rows a minibatch reads
together sit together in memory, without changing what is sampled.

The renumbered graph keeps every row and the order of neighbours inside it; only ids change
(row v_new = row order[v_new] of the original, neighbour ids mapped through perm). The
sampler is told the mapping (Sampler.set_reorder) and keys its Philox draws on the
original ids, so the sample is bit-identical to sampling the original graph, and training
cannot change. What changes is which cache lines the reads land on.

Methods (order[new id] = original id):
  degree   all nodes by in-degree, descending: hubs form a dense prefix of indptr and indices
           (the hub cache, D23 kHubHints, relies on this hub-first layout)
  hubsort  only the hubs (in-degree above the mean) move to the front, sorted by degree; the
           rest keep their original relative order, and with it any locality the original
           numbering had (Balaji & Lucia, "When is graph reordering an optimization?", 2018)
  rcm      reverse Cuthill-McKee (scipy): a BFS order that gives neighbours nearby ids;
           memory ~ 2-3x the edge array, so meant for the laptop-scale graphs, not Twitter
  random   a random permutation: destroys all locality; the control that shows how much the
           original numbering already helps

Usage: python -m gnnsample.reorder --graph products --method degree
  writes data/csc/products@degree/ (indptr, indices, order.int32.bin, meta.json);
  graph.load("products@degree") then returns it with the original graph's extras.
"""

import argparse
import json
import time

import numpy as np

METHODS = ("degree", "hubsort", "rcm", "random")


def compute_order(indptr, indices, method, symmetric=False, seed=0):
    """Returns order (int32): order[new id] = original id."""
    n = len(indptr) - 1
    deg = np.diff(np.asarray(indptr))
    if method == "degree":
        order = np.argsort(-deg, kind="stable")
    elif method == "hubsort":
        hub = deg > deg.mean()
        hubs = np.flatnonzero(hub)
        order = np.concatenate([hubs[np.argsort(-deg[hubs], kind="stable")], np.flatnonzero(~hub)])
    elif method == "rcm":
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        a = csr_matrix((np.ones(len(indices), dtype=np.int8), indices, indptr), shape=(n, n))
        order = reverse_cuthill_mckee(a, symmetric_mode=symmetric)
    elif method == "random":
        order = np.random.default_rng(seed).permutation(n)
    else:
        raise ValueError(f"method must be one of {METHODS}")
    return np.ascontiguousarray(order, dtype=np.int32)


def inverse(order):
    """perm[original id] = new id."""
    perm = np.empty(len(order), dtype=np.int32)
    perm[order] = np.arange(len(order), dtype=np.int32)
    return perm


def permute_csc(indptr, indices, order, out_indices=None, chunk_edges=1 << 24):
    """Renumbered CSC: row v of the result is row order[v] of the input, neighbours in the
    same order, ids mapped through inverse(order). Works in chunks of rows so that a
    memory-mapped out_indices can be filled for graphs larger than RAM headroom.
    Returns (indptr, indices)."""
    indptr = np.asarray(indptr, dtype=np.int64)
    n = len(order)
    perm = inverse(order)
    deg = (indptr[1:] - indptr[:-1])[order]
    new_indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=new_indptr[1:])
    if out_indices is None:
        out_indices = np.empty(int(new_indptr[-1]), dtype=np.int32)
    lo = 0
    while lo < n:
        hi = int(np.searchsorted(new_indptr, new_indptr[lo] + chunk_edges, side="right")) - 1
        hi = min(max(hi, lo + 1), n)
        lens = deg[lo:hi]
        base = new_indptr[lo:hi] - new_indptr[lo]  # each row's offset inside the chunk
        src = np.repeat(indptr[order[lo:hi]] - base, lens) + np.arange(int(lens.sum()))
        out_indices[new_indptr[lo]:new_indptr[hi]] = perm[indices[src]]
        lo = hi
    return new_indptr, out_indices


def pos_shift(orig_indptr, new_indptr, order):
    """Per new id: original CSC position = position in the renumbered graph + shift."""
    return np.ascontiguousarray(np.asarray(orig_indptr, dtype=np.int64)[order] - new_indptr[:-1])


def hub_count(indptr, budget_bytes):
    """Largest h such that the first h rows of a hub-first graph (their indptr entries and
    neighbour lists) fit in budget_bytes; ids below h are then the hub set."""
    indptr = np.asarray(indptr, dtype=np.int64)
    cost = 8 * np.arange(len(indptr), dtype=np.int64) + 4 * indptr
    return int(np.searchsorted(cost, budget_bytes, side="right")) - 1


def write_reordered(name, method, chunk_edges=1 << 24):
    """data/csc/<name> -> data/csc/<name>@<method>; returns the output directory."""
    from .graph import DATA, load

    g = load(name, extras=False, mmap=True)
    t0 = time.perf_counter()
    order = compute_order(g.indptr, g.indices, method, symmetric=not g.meta.get("directed", True))
    t_order = time.perf_counter() - t0
    out = DATA / f"{name}@{method}"
    out.mkdir(exist_ok=True)
    m = len(g.indices)
    dst = np.memmap(out / "indices.int32.bin", dtype=np.int32, mode="w+", shape=(m,))
    new_indptr, _ = permute_csc(g.indptr, g.indices, order, out_indices=dst,
                                chunk_edges=chunk_edges)
    dst.flush()
    del dst
    new_indptr.tofile(out / "indptr.int64.bin")
    order.tofile(out / "order.int32.bin")
    meta = dict(g.meta, name=f"{name}@{method}", base=name, reorder=method,
                order_seconds=round(t_order, 2),
                permute_seconds=round(time.perf_counter() - t0 - t_order, 2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="products")
    p.add_argument("--method", default="degree", choices=METHODS)
    args = p.parse_args()
    out = write_reordered(args.graph, args.method)
    print(f"wrote {out}")
    print((out / "meta.json").read_text())


if __name__ == "__main__":
    main()
