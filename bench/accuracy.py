"""Accuracy parity: does GraphSAGE trained with our sampler match PyG-trained accuracy?

One run = one (sampler, seed). Everything except the sampler is identical per seed: model
initialisation, seed order per epoch, optimiser, hyperparameters. Both models are scored by
the same evaluator (pyg-lib, the reference implementation), so a sampler bug could only
show up through training.

  ogbn-products, GraphSAGE 3x256, fanout [15,10,5], batch 1024, Adam lr 1e-3, trimmed
  layers; val accuracy (full val set) after every epoch, test accuracy on a fixed random
  50 K subset of the 2.2 M test nodes at the end (sampled inference, same fanouts).
  pyg: DataLoader with 2 workers (pyg-lib + index_select), pinned non-blocking copies
  ours: SYCL sampler + GPU gather (gnnsample.loader.training_sampler, the default batched
        sampler since D24; minibatches are bit-identical to the earlier single sampler)

Writes results/accuracy/<sampler>_seed<k>.json. Aggregate: bench/accuracy_report.py.
Usage: python bench/accuracy.py --sampler ours --seed 0 [--epochs 5]
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
BATCH = 1024
OUT = Path(__file__).resolve().parents[1] / "results" / "accuracy"


class PygTrainBatches(torch.utils.data.Dataset):
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


@torch.no_grad()
def evaluate(model, rowptr, col, x, y, nodes, device):
    """Sampled inference with pyg-lib (same for every model)."""
    model.eval()
    correct = 0
    for i in range(0, len(nodes), BATCH):
        seeds = nodes[i:i + BATCH]
        row, colx, node, _, n_nodes, n_edges = neighbor_sample(rowptr, col, seeds, FANOUTS,
                                                               csc=True)
        src, dst = orient(row, colx, n_edges, len(seeds))
        adjs = [(e.to(device), n) for e, n in
                layer_adjs(src, dst, list(n_nodes), list(n_edges), trim=True)]
        out = model(torch.index_select(x, 0, node).to(device), adjs)[: len(seeds)]
        correct += int((out.argmax(-1).cpu() == y[seeds]).sum())
    model.train()
    return correct / len(nodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sampler", choices=["ours", "pyg"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-batches", type=int, default=0, help="per epoch; 0 = all (smoke test)")
    p.add_argument("--threads", type=int, default=7)
    p.add_argument("--out-dir", default=None, help="write the run JSON here instead")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("xpu")

    g = load("products")
    x = torch.from_numpy(np.asarray(g.extras["feat"]))
    y = torch.from_numpy(np.asarray(g.extras["label"]))
    train = np.asarray(g.extras["train_idx"])
    val = torch.from_numpy(np.asarray(g.extras["valid_idx"]))
    test_all = np.asarray(g.extras["test_idx"])
    test = torch.from_numpy(np.sort(np.random.default_rng(0).choice(test_all, 50_000,
                                                                    replace=False)))
    rowptr = torch.from_numpy(g.indptr)
    col = torch.from_numpy(g.indices.astype(np.int64))

    # Identical seed order per (seed, epoch) for both samplers.
    rng = np.random.default_rng(1000 + args.seed)
    epochs = []
    for _ in range(args.epochs):
        perm = rng.permutation(train)
        b = [perm[i:i + BATCH] for i in range(0, len(perm), BATCH)]
        epochs.append(b[: args.max_batches] if args.max_batches else b)

    if args.sampler == "pyg":
        flat = [torch.from_numpy(b).long() for ep in epochs for b in ep]
        ds = PygTrainBatches(rowptr, col, x, y, flat)
        loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2,
                                             pin_memory=True, prefetch_factor=2)
        it = iter(loader)  # fork the workers before the GPU runtime starts
    else:
        from gnnsample.loader import training_sampler

        gs = training_sampler(g.indptr, g.indices, FANOUTS, BATCH, feat=x.numpy())

    torch.manual_seed(args.seed)
    model = SAGE(x.shape[1], 256, int(y.max()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y_dev = y.to(device)

    log = dict(sampler=args.sampler, seed=args.seed, epochs=[], fanouts=FANOUTS, batch=BATCH,
               lr=1e-3, hidden=256, layers=3)
    for ep, batches in enumerate(epochs):
        t0 = time.perf_counter()
        losses = []
        if args.sampler != "pyg":
            ours_it = gs.iterate_full(batches, seed_of=lambda i, ep=ep: (args.seed << 20) |
                                      (ep << 12) | i)
        for i, seeds in enumerate(batches):
            if args.sampler == "pyg":
                xb, yb, ei, n_nodes, n_edges = next(it)
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                ei = ei.to(device, non_blocking=True)
            else:
                r = next(ours_it)
                node, ei, xb = r["node"], r["edge_index"], r["x"]
                n_nodes, n_edges = r["num_sampled_nodes"], r["num_sampled_edges"]
                yb = y_dev[node[: len(seeds)]]
            adjs = layer_adjs(ei[0], ei[1], n_nodes, n_edges, trim=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb, adjs)[: len(seeds)], yb)
            loss.backward()
            opt.step()
            losses.append(float(loss))
            if i % 50 == 0:
                print(f"  epoch {ep} batch {i}/{len(batches)} loss {losses[-1]:.4f}",
                      flush=True)
        train_s = time.perf_counter() - t0
        val_acc = evaluate(model, rowptr, col, x, y, val, device)
        log["epochs"].append(dict(epoch=ep, loss=float(np.mean(losses)), val_acc=val_acc,
                                  train_s=train_s))
        print(f"epoch {ep}: loss {np.mean(losses):.4f} val {val_acc:.4f} "
              f"train {train_s:.0f} s", flush=True)
    log["test_acc_50k"] = evaluate(model, rowptr, col, x, y, test, device)
    print(f"test (50 K subset): {log['test_acc_50k']:.4f}", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.max_batches else ""
    (out_dir / f"{args.sampler}_seed{args.seed}{suffix}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
