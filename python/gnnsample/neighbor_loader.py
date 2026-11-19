"""Drop-in replacement for `torch_geometric.loader.NeighborLoader` (homogeneous graphs).

    from torch_geometric.loader import NeighborLoader   # before
    from gnnsample import NeighborLoader                # after

Sampling and feature gathering run with the SYCL sampler on the Intel GPU (`device="xpu"`) or
on the CPU (`device="cpu"`); `device="auto"` (default) uses the XPU when PyTorch has one. Each
minibatch arrives as a `torch_geometric.data.Data` already on that device, with the same
attributes PyG's loader produces: `x`, `edge_index` (neighbour -> target, local ids), `y`,
`n_id` (global ids; the first `batch_size` are the seeds), `batch_size`, `input_id`,
`num_sampled_nodes`, `num_sampled_edges` (so PyG's layer trimming works unchanged), and
`e_id` when `return_e_id=True`.

Semantics match PyG's default (`replace=False`, `subgraph_type="directional"`,
`disjoint=False`): each hop expands only the nodes the previous hop discovered, picks up to
k in-neighbours without replacement, and numbers nodes by first appearance. The random
draws differ from pyg-lib's, so individual minibatches differ; accuracy parity is shown in
docs/decisions.md D15.

Sampler: `sampler="batched"` (default, D24) samples K (`K=8`) minibatches per GPU launch;
`sampler="single"` samples one per launch (the earlier pipeline, and the only one that
takes custom `options` or fanouts above 16; with those, the default falls back to it with
a warning). Both give bit-identical minibatches for the same seed.

Differences from PyG, by design:
  - with `prefetch=True` (default) the next minibatches are prepared while one trains, and
    a minibatch's tensors are reused two minibatches later; keep a copy (`.clone()`) if you
    need them longer, or pass `prefetch=False`.
  - `e_id` is opt-in (`return_e_id=True`): it keeps an int64 permutation per edge on the
    output device (1 GB for ogbn-products).
  - worker/pinning arguments of the PyTorch DataLoader are accepted and ignored: there are
    no worker processes.
  - not supported (raises): heterogeneous graphs, `replace=True`, temporal sampling,
    `disjoint=True`, `transform`, edge weights, -1 (all neighbours) or fanouts above 64.
"""


import numpy as np
import torch

import warnings

from .loader import (BATCHED_MAX_FANOUT, DEFAULT_OPTIONS, DEFAULT_SAMPLER, resolve_device,
                     training_sampler)

_IGNORED = {"num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
            "filter_per_worker", "is_sorted", "directed", "worker_init_fn", "timeout"}
_UNSUPPORTED = {"input_time", "time_attr", "temporal_strategy", "weight_attr", "transform",
                "transform_sampler_output", "neighbor_sampler", "collate_fn", "batch_sampler"}


class NeighborLoader:
    def __init__(self, data, num_neighbors, input_nodes=None, batch_size=1, shuffle=False,
                 drop_last=False, replace=False, subgraph_type="directional", disjoint=False,
                 return_e_id=False, prefetch=True, seed=0, options=DEFAULT_OPTIONS,
                 device="auto", sampler=DEFAULT_SAMPLER, K=8, **kwargs):
        from torch_geometric.data import Data

        if not isinstance(data, Data):
            raise NotImplementedError("gnnsample.NeighborLoader supports homogeneous "
                                      "torch_geometric.data.Data only")
        bad = sorted(k for k, v in kwargs.items() if k in _UNSUPPORTED and v is not None)
        if bad or replace or disjoint or subgraph_type not in ("directional", None):
            raise NotImplementedError(
                f"not supported by gnnsample.NeighborLoader: "
                f"{bad + [n for n, f in (('replace', replace), ('disjoint', disjoint)) if f]}"
                f"{'' if subgraph_type in ('directional', None) else ' subgraph_type'}")
        unknown = sorted(set(kwargs) - _IGNORED - _UNSUPPORTED)
        if unknown:
            raise TypeError(f"unexpected arguments: {unknown}")
        num_neighbors = list(num_neighbors)
        if any((not isinstance(k, int)) or k < 1 or k > 64 for k in num_neighbors):
            raise NotImplementedError("num_neighbors must be integers in [1, 64] "
                                      "(-1 = all neighbours is not supported)")

        self.data, self.batch_size, self.shuffle, self.drop_last = data, batch_size, shuffle, drop_last
        self.seed, self.prefetch, self.return_e_id = seed, prefetch, return_e_id
        self.num_nodes = data.num_nodes
        self.epoch = 0
        self.device = resolve_device(device)

        # In-neighbour CSC (what the sampler reads): sort edges by target, keep source ids.
        src, dst = data.edge_index[0].cpu(), data.edge_index[1].cpu()
        perm = torch.argsort(dst * self.num_nodes + src)
        indptr = torch.zeros(self.num_nodes + 1, dtype=torch.int64)
        indptr[1:] = torch.cumsum(torch.bincount(dst, minlength=self.num_nodes), 0)
        indices = src[perm].to(torch.int32)

        x = getattr(data, "x", None)
        self.dim = None
        feat = None
        if x is not None:
            x = x.detach().cpu().float()
            self.dim = x.shape[1]
            pad = (-self.dim) % 4  # the device gather moves 16-byte chunks
            feat = torch.nn.functional.pad(x, (0, pad)).numpy() if pad else x.numpy()
        if sampler not in ("batched", "single"):
            raise ValueError("sampler must be 'batched' or 'single'")
        if sampler == "batched" and (options != DEFAULT_OPTIONS or
                                     max(num_neighbors) > BATCHED_MAX_FANOUT):
            warnings.warn("gnnsample.NeighborLoader: custom options or fanouts above "
                          f"{BATCHED_MAX_FANOUT} need sampler='single'; using it")
            sampler = "single"
        self.sampler_kind = sampler
        self.sampler = training_sampler(indptr.numpy(), indices.numpy(), num_neighbors,
                                        batch_size, feat=feat, device=self.device,
                                        sampler=sampler, prefetch=prefetch, K=K,
                                        with_edge_pos=return_e_id, options=options)
        self.feat_cols = None if feat is None else feat.shape[1]

        dev = self.device
        y = getattr(data, "y", None)
        self.y = y.to(dev) if y is not None else None
        self.e_perm = perm.to(dev) if return_e_id else None  # CSC position -> edge_index column

        if input_nodes is None:
            self.input_nodes = torch.arange(self.num_nodes)
        elif input_nodes.dtype == torch.bool:
            self.input_nodes = input_nodes.cpu().nonzero().view(-1)
        else:
            self.input_nodes = input_nodes.cpu().long()

    def __len__(self):
        n = len(self.input_nodes)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def _batches(self):
        # input_id is the seeds' position in input_nodes (as in PyG), also when shuffled.
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
            idx = torch.randperm(len(self.input_nodes), generator=g)
        else:
            idx = torch.arange(len(self.input_nodes))
        out = []
        for i in range(len(self)):
            sl = idx[i * self.batch_size:(i + 1) * self.batch_size]
            out.append((self.input_nodes[sl].numpy().astype(np.int32), sl))
        return out

    def __iter__(self):
        from torch_geometric.data import Data

        batches = self._batches()
        epoch = self.epoch
        self.epoch += 1
        seed_of = lambda i: (self.seed << 40) ^ (epoch << 20) ^ i  # noqa: E731
        results = self.sampler.iterate_full([b for b, _ in batches], seed_of)
        for r, (seeds, input_pos) in zip(results, batches):  # results first: resumed to its end
            node = r["node"]
            batch = Data(edge_index=r["edge_index"], n_id=node, batch_size=len(seeds),
                         num_sampled_nodes=list(r["num_sampled_nodes"]),
                         num_sampled_edges=list(r["num_sampled_edges"]),
                         input_id=input_pos)
            batch.num_nodes = len(node)
            if r["x"] is not None:
                batch.x = r["x"] if self.feat_cols == self.dim else r["x"][:, : self.dim]
            if self.y is not None:
                batch.y = self.y[node]
            if self.return_e_id:
                batch.e_id = self.e_perm[r["edge_pos"]]
            yield batch

    def __repr__(self):
        return (f"gnnsample.NeighborLoader(nodes={self.num_nodes}, batch_size={self.batch_size}, "
                f"batches={len(self)}, device={self.device.type}, sampler={self.sampler_kind}, "
                f"prefetch={self.prefetch})")
