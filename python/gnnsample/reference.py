"""Reference multi-hop neighbor sampler (NumPy). It defines the semantics every fast
implementation must reproduce bit-for-bit.

Semantics (match PyG's NeighborSampler, disjoint=False, replace=False):
  - Hop h expands only the nodes first discovered at hop h-1 (hop 0 expands the seeds).
  - For a node v with in-degree d and fanout k:
      d <= k: take all d neighbors in CSC order.
      d >  k: Robert Floyd's algorithm. For i in 0..k-1 with j = d-k+i, draw
              t = bounded(philox(seed, hop, v, i), j+1); pick t if unseen, else j.
              Picks are kept in draw order.
  - Nodes get local IDs in order of first appearance: seeds first, then each hop's new
    nodes in the order their sampled edges were produced.
  - Output edges point neighbor -> target: row = local id of the neighbor (source of the
    message), col = local id of the target, edge = global CSC position.
"""

from dataclasses import dataclass

import numpy as np

from .philox import bounded, draw_u32


@dataclass
class Sample:
    node: np.ndarray            # global ids, local id = position
    row: np.ndarray             # local id of sampled neighbor
    col: np.ndarray             # local id of target node
    edge: np.ndarray            # global CSC position of each sampled edge
    num_sampled_nodes: list     # per hop, starting with the seeds
    num_sampled_edges: list     # per hop


def sample_hop(indptr, indices, frontier, k, seed, hop):
    """Returns (targets, edge positions) for one hop; targets repeat per sampled edge."""
    start = indptr[frontier]
    deg = indptr[frontier + 1] - start
    take = np.minimum(deg, k)

    # Positions within each node's neighbor list, laid out per frontier node in order.
    out_off = np.zeros(len(frontier) + 1, dtype=np.int64)
    np.cumsum(take, out=out_off[1:])
    pos = np.empty(out_off[-1], dtype=np.int64)

    small = deg <= k
    if small.any():
        # All neighbors: position i of node f lands at out_off[f] + i.
        f = np.repeat(np.flatnonzero(small), take[small])
        within = np.arange(len(f)) - np.repeat(np.cumsum(take[small]) - take[small], take[small])
        pos[out_off[f] + within] = within

    big = np.flatnonzero(~small)
    if len(big):
        d = deg[big]
        picks = np.empty((len(big), k), dtype=np.int64)
        for i in range(k):
            j = d - k + i
            t = bounded(draw_u32(seed, hop, frontier[big], i), j + 1)
            seen = (picks[:, :i] == t[:, None]).any(axis=1)
            picks[:, i] = np.where(seen, j, t)
        pos[(out_off[big][:, None] + np.arange(k)).ravel()] = picks.ravel()

    targets = np.repeat(np.arange(len(frontier)), take)
    return targets, start[targets] + pos


def sample(indptr, indices, seeds, fanouts, seed=0) -> Sample:
    indptr = np.asarray(indptr, dtype=np.int64)
    seeds = np.asarray(seeds, dtype=np.int64)
    nodes = [seeds]
    lookup = {int(v): i for i, v in enumerate(seeds)}
    frontier_lo = 0
    rows, cols, edges = [], [], []
    n_nodes, n_edges = [len(seeds)], []

    for hop, k in enumerate(fanouts):
        all_nodes = np.concatenate(nodes)
        frontier = all_nodes[frontier_lo:]
        tgt, epos = sample_hop(indptr, indices, frontier, k, seed, hop)
        nbr = indices[epos].astype(np.int64)

        local = np.empty(len(nbr), dtype=np.int64)
        new = []
        for i, v in enumerate(nbr.tolist()):
            if v not in lookup:
                lookup[v] = len(lookup)
                new.append(v)
            local[i] = lookup[v]
        new = np.array(new, dtype=np.int64)

        rows.append(local)
        cols.append(frontier_lo + tgt)
        edges.append(epos)
        n_edges.append(len(epos))
        n_nodes.append(len(new))
        frontier_lo = len(all_nodes)
        nodes.append(new)

    return Sample(np.concatenate(nodes), np.concatenate(rows), np.concatenate(cols),
                  np.concatenate(edges), n_nodes, n_edges)
