"""Steady-state training throughput: our GPU pipeline vs PyG at its best (overlapped).

profile_pyg.py times stages in a serial loop. Real PyG overlaps CPU sampling + gather with
GPU training through NeighborLoader worker processes, so the fair end-to-end baseline
is the wall time per minibatch with workers (each worker samples, gathers into shared
memory, and the main process copies from pinned memory and trains).

  pyg-wN   DataLoader with N workers, each item = pyg-lib neighbor_sample + index_select
           (what NeighborLoader does), pin_memory=True, non_blocking copies
  ours     SYCL sampler + GPU gather from host USM, serial (sampling uses the same iGPU)
  ours-overlap   single-minibatch sampler, minibatch i+1 prepared while i trains (D16)
  ours-batched   the default batched sampler (D24): K=8 minibatches per launch, overlapped
Model and step identical to profile_pyg.py (GraphSAGE 3x256, [15,10,5], trimmed).
Reports median and mean ms per minibatch after warm-up.

Usage: for c in ours pyg-w0 pyg-w4; do python bench/train_e2e.py --config $c; done
"""

import argparse
import json
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


class PygMinibatches(torch.utils.data.Dataset):
    def __init__(self, rowptr, col, x, y, batches):
        self.rowptr, self.col, self.x, self.y, self.batches = rowptr, col, x, y, batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        seeds = self.batches[i]
        row, colx, node, _, n_nodes, n_edges = neighbor_sample(self.rowptr, self.col, seeds,
                                                               FANOUTS, csc=True)
        src, dst = orient(row, colx, n_edges, len(seeds))
        return (torch.index_select(self.x, 0, node), self.y[node[: len(seeds)]],
                torch.stack([src, dst]), list(n_nodes), list(n_edges))


def train_step(model, opt, xb, yb, adjs, bs):
    opt.zero_grad()
    F.cross_entropy(model(xb, adjs)[:bs], yb).backward()
    opt.step()


def run_pyg(g, x, y, batches, workers, make_model, device, warmup):
    rowptr = torch.from_numpy(g.indptr)
    col = torch.from_numpy(g.indices.astype(np.int64))
    ds = PygMinibatches(rowptr, col, x, y, [torch.from_numpy(b).long() for b in batches])
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers,
                                         pin_memory=True, prefetch_factor=2 if workers else None,
                                         persistent_workers=False)
    # Start (fork) the workers before anything touches the GPU: forking after the SYCL
    # runtime is initialised deadlocks the workers.
    it = iter(loader)
    model, opt = make_model()
    times, t_last = [], None
    for i, (xb, yb, ei, n_nodes, n_edges) in enumerate(it):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        ei = ei.to(device, non_blocking=True)
        adjs = layer_adjs(ei[0], ei[1], n_nodes, n_edges, trim=True)
        train_step(model, opt, xb, yb, adjs, len(yb))
        torch.xpu.synchronize()
        now = time.perf_counter()
        if i % 10 == 0:
            print(f"  pyg batch {i}/{len(batches)}", flush=True)
        if t_last is not None and i > warmup:
            times.append(now - t_last)
        t_last = now
    return times


def run_ours(g, x, y, batches, make_model, device, warmup):
    model, opt = make_model()
    from gnnsample.loader import GpuSampler

    gs = GpuSampler(g.indptr, g.indices, FANOUTS, max(map(len, batches)), feat=x.numpy())
    y_dev = y.to(device)
    times, t_last = [], None
    for i, b in enumerate(batches):
        node, ei, xb, n_nodes, n_edges = gs.sample(b, seed=i)
        adjs = layer_adjs(ei[0], ei[1], n_nodes, n_edges, trim=True)
        train_step(model, opt, xb, y_dev[node[: len(b)]], adjs, len(b))
        torch.xpu.synchronize()
        now = time.perf_counter()
        if i % 10 == 0:
            print(f"  ours batch {i}/{len(batches)}", flush=True)
        if t_last is not None and i > warmup:
            times.append(now - t_last)
        t_last = now
    return times


def run_ours_overlap(g, x, y, batches, make_model, device, warmup):
    from gnnsample.loader import PrefetchingGpuSampler

    model, opt = make_model()
    gs = PrefetchingGpuSampler(g.indptr, g.indices, FANOUTS, max(map(len, batches)),
                               feat=x.numpy())
    y_dev = y.to(device)
    times, t_last = [], None
    for i, (node, ei, xb, n_nodes, n_edges) in enumerate(gs.iterate(batches)):
        bs = len(batches[i])
        adjs = layer_adjs(ei[0], ei[1], n_nodes, n_edges, trim=True)
        train_step(model, opt, xb, y_dev[node[:bs]], adjs, bs)
        torch.xpu.synchronize()
        now = time.perf_counter()
        if t_last is not None and i > warmup:
            times.append(now - t_last)
        t_last = now
        if i % 10 == 0:
            print(f"  ours-overlap batch {i}/{len(batches)}", flush=True)
    return times


def run_ours_batched(g, x, y, batches, make_model, device, warmup):
    from gnnsample.loader import training_sampler

    model, opt = make_model()
    gs = training_sampler(g.indptr, g.indices, FANOUTS, max(map(len, batches)), feat=x.numpy())
    y_dev = y.to(device)
    times, t_last = [], None
    for i, (node, ei, xb, n_nodes, n_edges) in enumerate(gs.iterate(batches)):
        bs = len(batches[i])
        adjs = layer_adjs(ei[0], ei[1], n_nodes, n_edges, trim=True)
        train_step(model, opt, xb, y_dev[node[:bs]], adjs, bs)
        torch.xpu.synchronize()
        now = time.perf_counter()
        if t_last is not None and i > warmup:
            times.append(now - t_last)
        t_last = now
        if i % 10 == 0:
            print(f"  ours-batched batch {i}/{len(batches)}", flush=True)
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="products")
    p.add_argument("--config", required=True, help="ours, ours-overlap, ours-batched or pyg-wN (one per process: "
                   "DataLoader workers must fork before the GPU runtime starts)")
    p.add_argument("--batches", type=int, default=60)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--threads", type=int, default=7)
    p.add_argument("--json-out", default=None, help="append the result here instead")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("xpu")

    g = load(args.graph)
    x = torch.from_numpy(np.asarray(g.extras["feat"]))
    y = torch.from_numpy(np.asarray(g.extras["label"]))
    seeds = np.random.default_rng(0).permutation(np.asarray(g.train_seeds()))
    batches = [seeds[i * 1024:(i + 1) * 1024] for i in range(args.batches)]

    def make_model():
        torch.manual_seed(0)
        model = SAGE(x.shape[1], 256, int(y.max()) + 1).to(device)
        return model, torch.optim.Adam(model.parameters(), lr=1e-3)

    rows = []
    for cfg in [args.config]:
        if cfg == "ours":
            t = run_ours(g, x, y, batches, make_model, device, args.warmup)
        elif cfg == "ours-overlap":
            t = run_ours_overlap(g, x, y, batches, make_model, device, args.warmup)
        elif cfg == "ours-batched":
            t = run_ours_batched(g, x, y, batches, make_model, device, args.warmup)
        else:
            t = run_pyg(g, x, y, batches, int(cfg.split("-w")[1]), make_model, device,
                        args.warmup)
        rows.append(dict(config=cfg, median_ms=float(np.median(t) * 1e3),
                         mean_ms=float(np.mean(t) * 1e3), n=len(t)))
        print(f"{cfg:8} median {rows[-1]['median_ms']:6.1f} ms/step  "
              f"mean {rows[-1]['mean_ms']:6.1f}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(args.json_out or RESULTS / "train_e2e.jsonl", "a") as f:
        f.write(json.dumps(dict(graph=args.graph, batches=args.batches, rows=rows)) + "\n")


if __name__ == "__main__":
    main()
