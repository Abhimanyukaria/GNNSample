"""Why does a GraphSAGE train step take ~1.7 s on the iGPU? (follow-up to D9)

Samples a fixed set of minibatches once (pyg-lib, ogbn-products, [15,10,5], batch 1024),
keeps them on the device, then times only forward + backward + optimizer step under:
  scatter        PyG SAGEConv as in profile_pyg.py (edge-wise gather + scatter-mean)
  scatter-trim   same, with PyG layer trimming (layer l only computes the nodes that
                 later layers need; num_sampled_nodes/edges_per_hop)
  spmm           aggregation as a sparse CSR matmul (torch.sparse) instead of scatter
  spmm-trim      both
Each variant is run on the iGPU (xpu) and on the CPU. The first variant also reports a
forward/backward split and the top kernels from torch.profiler.

Usage: python bench/train_step.py [--batches 20] [--variants ...] [--devices xpu,cpu]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pyg_lib.sampler import neighbor_sample
from torch_geometric.nn import SAGEConv

from gnnsample.graph import load

FANOUTS = [15, 10, 5]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()


class SAGE(torch.nn.Module):
    """3-layer GraphSAGE. With `trim`, layer l runs only on the edges/nodes still needed
    (same idea as PyG's trim_to_layer): the last layer needs only edges into the seeds."""

    def __init__(self, d_in, d_hidden, d_out, layers=3):
        super().__init__()
        dims = [d_in] + [d_hidden] * (layers - 1) + [d_out]
        self.convs = torch.nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x, adjs):
        # adjs[l]: aggregation input for layer l and the number of target rows it produces
        for l, conv in enumerate(self.convs):
            adj, n_dst = adjs[l]
            x = conv((x, x[:n_dst]), adj)
            if l < len(self.convs) - 1:
                x = F.relu(x)
        return x


def make_batches(g, n, batch, device):
    rowptr = torch.from_numpy(g.indptr)
    col = torch.from_numpy(g.indices.astype(np.int64))
    x = torch.from_numpy(np.asarray(g.extras["feat"]))
    y = torch.from_numpy(np.asarray(g.extras["label"]))
    train = torch.from_numpy(np.asarray(g.extras["train_idx"]))
    perm = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(0))]
    out = []
    for b in range(n):
        seeds = perm[b * batch:(b + 1) * batch]
        row, colx, node, _, n_nodes, n_edges = neighbor_sample(rowptr, col, seeds, FANOUTS,
                                                               csc=True)
        # Orientation check: hop-0 edges must point into the seeds (local ids < batch).
        e0 = n_edges[0]
        if colx[:e0].max() < len(seeds):
            src, dst = row, colx
        else:
            src, dst = colx, row
        assert dst[:e0].max() < len(seeds)
        out.append(dict(x=x[node], y=y[node[: len(seeds)]], src=src, dst=dst,
                        n_nodes=list(n_nodes), n_edges=list(n_edges)))
    return out


def build_adjs(b, mode, trim, device, layers=3):
    """Per layer (adj, n_dst). Edge hops are ordered: hop h's edges target the nodes added
    at hop h-1. Layer l (0-based) of an L-layer model consumes hops 0..L-1-l when trimmed."""
    src, dst = b["src"], b["dst"]
    n_total = sum(b["n_nodes"])
    cum_e = np.cumsum([0] + b["n_edges"])
    cum_n = np.cumsum([0] + b["n_nodes"])
    adjs = []
    for l in range(layers):
        if trim:
            keep_hops = layers - l              # layer 0 uses all hops, last uses hop 0
            e_hi = int(cum_e[keep_hops])
            n_dst = int(cum_n[keep_hops])       # nodes that still need an output
            n_src = int(cum_n[keep_hops + 1])   # rows of this layer's input
        else:
            e_hi, n_dst, n_src = len(src), n_total, n_total
        s, d = src[:e_hi], dst[:e_hi]
        if mode == "scatter":
            adj = torch.stack([s, d]).to(device)
        else:  # spmm: adj_t[dst, src] in CSR, mean aggregation via row normalisation
            order = torch.argsort(d * n_src + s)
            s, d = s[order], d[order]
            deg = torch.bincount(d, minlength=n_dst).clamp(min=1)
            crow = torch.zeros(n_dst + 1, dtype=torch.long)
            crow[1:] = torch.cumsum(torch.bincount(d, minlength=n_dst), 0)
            val = (1.0 / deg[d]).float()
            adj = torch.sparse_csr_tensor(crow, s, val, (n_dst, n_src)).to(device)
        adjs.append((adj, n_dst))
    return adjs


class MeanSAGE(torch.nn.Module):
    """Same math as SAGEConv(aggr=mean), with aggregation done as adj_t @ x."""

    def __init__(self, d_in, d_hidden, d_out, layers=3):
        super().__init__()
        dims = [d_in] + [d_hidden] * (layers - 1) + [d_out]
        self.lin_l = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.lin_r = torch.nn.ModuleList(torch.nn.Linear(a, b, bias=False)
                                         for a, b in zip(dims, dims[1:]))

    def forward(self, x, adjs):
        for l, (adj, n_dst) in enumerate(adjs):
            x = self.lin_l[l](adj @ x) + self.lin_r[l](x[:n_dst])
            if l < len(adjs) - 1:
                x = F.relu(x)
        return x


def time_variant(batches, mode, trim, device, d_in, n_cls, profile=False):
    torch.manual_seed(0)
    model = (SAGE if mode == "scatter" else MeanSAGE)(d_in, 256, n_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    prepared = [(b["x"].to(device), b["y"].to(device), build_adjs(b, mode, trim, device),
                 len(b["y"])) for b in batches]
    fwd, bwd, step = [], [], []
    for i, (x, y, adjs, bs) in enumerate(prepared):
        sync(device)
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, adjs)[:bs], y)
        sync(device)
        t1 = time.perf_counter()
        loss.backward()
        sync(device)
        t2 = time.perf_counter()
        opt.step()
        sync(device)
        t3 = time.perf_counter()
        if i >= 3:  # first batches warm up kernels/allocator
            fwd.append(t1 - t0)
            bwd.append(t2 - t1)
            step.append(t3 - t0)
    res = dict(mode=mode, trim=trim, device=device.type,
               step_s=float(np.median(step)), fwd_s=float(np.median(fwd)),
               bwd_s=float(np.median(bwd)))
    if profile:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "xpu":
            acts.append(torch.profiler.ProfilerActivity.XPU)
        x, y, adjs, bs = prepared[-1]
        with torch.profiler.profile(activities=acts) as prof:
            for _ in range(3):
                opt.zero_grad()
                F.cross_entropy(model(x, adjs)[:bs], y).backward()
                opt.step()
            sync(device)
        key = "self_xpu_time_total" if device.type == "xpu" else "self_cpu_time_total"
        table = prof.key_averages().table(sort_by=key, row_limit=12)
        res["profile_top"] = table
        print(table, flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--variants", default="scatter,scatter-trim,spmm,spmm-trim")
    p.add_argument("--devices", default="xpu,cpu")
    p.add_argument("--threads", type=int, default=14)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    g = load("products")
    batches = make_batches(g, args.batches, 1024, torch.device("cpu"))
    d_in, n_cls = batches[0]["x"].shape[1], int(np.asarray(g.extras["label"]).max()) + 1
    print(f"{len(batches)} batches, mean {np.mean([sum(b['n_nodes']) for b in batches]):,.0f} "
          f"nodes / {np.mean([len(b['src']) for b in batches]):,.0f} edges; per-hop nodes "
          f"{batches[0]['n_nodes']}, edges {batches[0]['n_edges']}", flush=True)

    rows = []
    for dev in args.devices.split(","):
        device = torch.device(dev)
        for v in args.variants.split(","):
            mode, trim = v.split("-")[0], v.endswith("-trim")
            try:
                r = time_variant(batches, mode, trim, device, d_in, n_cls,
                                 profile=(v == "scatter"))
            except Exception as e:  # e.g. sparse CSR matmul not supported on a device
                r = dict(mode=mode, trim=trim, device=dev, error=f"{type(e).__name__}: {e}")
            rows.append(r)
            print({k: v for k, v in r.items() if k != "profile_top"}, flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "train_step_products.json").write_text(json.dumps(dict(
        batches=args.batches, threads=args.threads, torch=torch.__version__, results=rows),
        indent=2))
    print(f"\n{'variant':16} {'device':6} {'step s':>8} {'fwd s':>8} {'bwd s':>8}")
    for r in rows:
        name = r["mode"] + ("-trim" if r["trim"] else "")
        if "error" in r:
            print(f"{name:16} {r['device']:6} {r['error'][:60]}")
        else:
            print(f"{name:16} {r['device']:6} {r['step_s']:8.3f} {r['fwd_s']:8.3f} "
                  f"{r['bwd_s']:8.3f}")


if __name__ == "__main__":
    main()
