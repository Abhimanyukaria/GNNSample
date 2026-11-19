"""Phase 1 / decision D9: where does a PyG GraphSAGE training step spend its time?

Per minibatch (fanout [15,10,5], 3 layers, hidden 256, batch 1024 seeds) we time:
  sample   pyg-lib neighbor_sample on the CSC (CPU; what NeighborLoader calls)
  gather   feature/label slicing for the sampled nodes (CPU)
  h2d      copy of features + edges to the device (on the iGPU this is a copy into
           device USM, even though the memory is physically shared)
  train    forward + backward + optimizer step on the device (layers trimmed to the
           nodes later layers need, unless --no-trim; see gnnsample/models.py)
Each stage is synchronised, so the split is exact for a serial (num_workers=0) loop.
Results go to results/profile_pyg_<graph>_<device>.json.

Usage: python bench/profile_pyg.py --graph products --device xpu [--batches 100]
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pyg_lib.sampler import neighbor_sample

from gnnsample.graph import load
from gnnsample.models import SAGE, layer_adjs, orient

FANOUTS = [15, 10, 5]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="products")
    p.add_argument("--device", default="xpu")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threads", type=int, default=torch.get_num_threads())
    p.add_argument("--no-trim", action="store_true", help="run every layer on all nodes")
    p.add_argument("--sampler", choices=["pyg", "ours"], default="pyg",
                   help="pyg: pyg-lib on the CPU + CPU gather + copy; ours: SYCL sampler and "
                        "GPU gather from host USM, tensors born on the device (D11)")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    g = load(args.graph)
    rowptr = torch.from_numpy(g.indptr)
    col = torch.from_numpy(g.indices.astype(np.int64))  # pyg-lib wants matching index dtypes
    if "feat" in g.extras:
        x = torch.from_numpy(np.asarray(g.extras["feat"]))
        y = torch.from_numpy(np.asarray(g.extras["label"]))
    else:  # gSampler convention for featureless graphs: random 128-d features
        x = torch.randn(g.num_nodes, 128)
        y = torch.randint(0, 16, (g.num_nodes,))
    n_classes = int(y.max()) + 1
    train = torch.from_numpy(np.asarray(g.train_seeds())).long()

    model = SAGE(x.shape[1], 256, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    perm = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(0))]
    if args.sampler == "ours":
        from gnnsample.loader import GpuSampler

        gs = GpuSampler(g.indptr, g.indices, FANOUTS, args.batch, feat=x.numpy())
        y_dev = y.to(device)

    stages = {k: [] for k in ("sample", "gather", "h2d", "train")}
    sizes = {"nodes": [], "edges": []}
    total = args.warmup + args.batches
    for b in range(total):
        seeds = perm[(b * args.batch) % len(perm):][: args.batch]
        if args.sampler == "ours":
            # sample: our sampler, results stay on the device
            # gather: node ids, edge_index and feature rows written straight into XPU tensors
            # h2d:    nothing crosses; labels and per-layer edge slices are built on device
            t0 = time.perf_counter()
            c = gs.s.run(seeds.numpy().astype(np.int32), b)
            t1 = time.perf_counter()
            node = torch.empty(c["num_nodes"], dtype=torch.long, device=device)
            edge_index = torch.empty(2, c["num_edges"], dtype=torch.long, device=device)
            xb = torch.empty(c["num_nodes"], x.shape[1], device=device)
            sync(device)
            gs.s.export_to(node.data_ptr(), edge_index.data_ptr(), xb.data_ptr())
            t2 = time.perf_counter()
            n_nodes, n_edges = c["num_sampled_nodes"], c["num_sampled_edges"]
            yb = y_dev[node[: len(seeds)]]
            adjs = layer_adjs(edge_index[0], edge_index[1], n_nodes, n_edges,
                              trim=not args.no_trim)
            sync(device)
            t3 = time.perf_counter()
            row = edge_index[0]
        else:
            t0 = time.perf_counter()
            row, colx, node, _, n_nodes, n_edges = neighbor_sample(rowptr, col, seeds,
                                                                   FANOUTS, csc=True)
            t1 = time.perf_counter()
            xb = torch.index_select(x, 0, node)  # what PyG's loader does
            yb = y[node[: len(seeds)]]
            src, dst = orient(row, colx, n_edges, len(seeds))
            adjs = layer_adjs(src, dst, n_nodes, n_edges, trim=not args.no_trim)
            t2 = time.perf_counter()
            xb, yb = xb.to(device), yb.to(device)
            adjs = [(e.to(device), n) for e, n in adjs]
            sync(device)
            t3 = time.perf_counter()
        opt.zero_grad()
        out = model(xb, adjs)[: len(seeds)]
        loss = F.cross_entropy(out, yb)
        loss.backward()
        opt.step()
        sync(device)
        t4 = time.perf_counter()
        if b >= args.warmup:
            for k, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                stages[k].append(dt)
            sizes["nodes"].append(len(node))
            sizes["edges"].append(row.numel())

    med = {k: float(np.median(v)) for k, v in stages.items()}
    step = sum(med.values())
    report = dict(
        graph=args.graph, sampler=args.sampler, device=str(device), threads=args.threads,
        trim=not args.no_trim, batch=args.batch,
        fanouts=FANOUTS, batches=args.batches,
        median_s=med, share={k: v / step for k, v in med.items()},
        iqr_s={k: [float(np.percentile(v, 25)), float(np.percentile(v, 75))]
               for k, v in stages.items()},
        mean_nodes=float(np.mean(sizes["nodes"])), mean_edges=float(np.mean(sizes["edges"])),
        feature_mb=float(np.mean(sizes["nodes"]) * x.shape[1] * 4 / 1e6),
        host=platform.processor() or platform.machine(), torch=torch.__version__,
        device_name=torch.xpu.get_device_name() if device.type == "xpu" else "cpu",
    )
    RESULTS.mkdir(exist_ok=True)
    tag = "" if args.no_trim else "_trim"
    tag += "" if args.sampler == "pyg" else "_ours"
    out = RESULTS / f"profile_pyg_{args.graph}_{device.type}{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
