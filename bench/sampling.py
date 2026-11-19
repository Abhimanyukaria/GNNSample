"""Sampling-only throughput: SYCL sampler vs pyg-lib (PyG's sampler).

Protocol (decision D8, gSampler convention): 6 epochs over the training seeds in batches of
1024, fanout [15,10,5]; report the mean epoch time of the last 5 epochs, plus sampled
edges/second. Each sampler gets the same CSC and the same seed batches.

pyg-lib samples one batch per call on one thread; NeighborLoader parallelises with worker
processes, so the PyG baseline sweeps num_workers and we compare against its best.
Our timing includes copying the result back to host NumPy arrays (conservative for us).

Usage: python bench/sampling.py --graph products [--samplers pyg,sycl-gpu,sycl-cpu]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gnnsample.graph import load

FANOUTS = [15, 10, 5]
RESULTS = Path(__file__).resolve().parents[1] / "results"


class PygBatches(torch.utils.data.Dataset):
    """One item = one sampled minibatch, so DataLoader workers sample in parallel."""

    def __init__(self, rowptr, col, batches):
        self.rowptr, self.col, self.batches = rowptr, col, batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        from pyg_lib.sampler import neighbor_sample

        row, _, _, _, _, _ = neighbor_sample(self.rowptr, self.col, self.batches[i], FANOUTS,
                                             csc=True)
        return row.numel()


def run_pyg(g, batches, workers, epochs):
    rowptr = torch.from_numpy(g.indptr)
    col = torch.from_numpy(g.indices.astype(np.int64))
    ds = PygBatches(rowptr, col, [torch.from_numpy(b).long() for b in batches])
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=workers,
                                         persistent_workers=workers > 0)
    times, edges = [], 0
    for _ in range(epochs):
        t0 = time.perf_counter()
        edges = sum(loader)
        times.append(time.perf_counter() - t0)
    return times, edges


def run_sycl(g, batches, device, epochs):
    from gnnsample import _C

    s = _C.Sampler(g.indptr, g.indices, FANOUTS, max(len(b) for b in batches), device)
    times, edges = [], 0
    for e in range(epochs):
        t0 = time.perf_counter()
        edges = 0
        for i, b in enumerate(batches):
            edges += int(sum(s.sample(b.astype(np.int32), e * len(batches) + i)["num_sampled_edges"]))
        times.append(time.perf_counter() - t0)
    return times, edges, s.device_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="products")
    p.add_argument("--samplers", default="pyg,sycl-gpu,sycl-cpu")
    p.add_argument("--workers", default="0,4,8,12")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=6)
    args = p.parse_args()

    g = load(args.graph, extras=False)
    train_idx = Path(__file__).resolve().parents[1] / f"data/csc/{g.name}/train_idx.npy"
    seeds = np.load(train_idx) if train_idx.exists() else g.train_seeds()
    rng = np.random.default_rng(0)
    seeds = rng.permutation(seeds)
    batches = [seeds[i:i + args.batch] for i in range(0, len(seeds), args.batch)]
    print(f"{g.name}: {g.num_nodes:,} nodes, {len(g.indices):,} edges, "
          f"{len(seeds):,} seeds, {len(batches)} batches/epoch", flush=True)

    rows = []
    for name in args.samplers.split(","):
        if name == "pyg":
            for w in map(int, args.workers.split(",")):
                times, edges = run_pyg(g, batches, w, args.epochs)
                rows.append(dict(sampler="pyg-lib", config=f"workers={w}", epoch_times=times,
                                 edges_per_epoch=edges))
                print(rows[-1]["sampler"], rows[-1]["config"],
                      f"{np.mean(times[1:]):.3f} s/epoch", flush=True)
        else:
            dev = name.split("-")[1]
            times, edges, dname = run_sycl(g, batches, dev, args.epochs)
            rows.append(dict(sampler="sycl-v1", config=dname, epoch_times=times,
                             edges_per_epoch=edges))
            print(rows[-1]["sampler"], dname, f"{np.mean(times[1:]):.3f} s/epoch", flush=True)

    for r in rows:
        t = r["epoch_times"][1:]
        r["epoch_s_mean"] = float(np.mean(t))
        r["epoch_s_std"] = float(np.std(t))
        r["medges_per_s"] = r["edges_per_epoch"] / r["epoch_s_mean"] / 1e6
    report = dict(graph=g.name, num_nodes=g.num_nodes, num_edges=int(len(g.indices)),
                  seeds=int(len(seeds)), batch=args.batch, fanouts=FANOUTS,
                  epochs=args.epochs, cpu_threads=torch.get_num_threads(), results=rows)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"sampling_{g.name}.json").write_text(json.dumps(report, indent=2))
    print(f"\n{'sampler':10} {'config':34} {'s/epoch':>9} {'M edges/s':>10}")
    for r in rows:
        print(f"{r['sampler']:10} {r['config'][:34]:34} {r['epoch_s_mean']:9.3f} "
              f"{r['medges_per_s']:10.2f}")


if __name__ == "__main__":
    main()
