"""A/B: sampling minibatch i+1 while training minibatch i (PrefetchingGpuSampler) vs the
serial pipeline (GpuSampler), same process, alternating blocks of minibatches so GPU-clock
drift hits both (decision D12).

Training step as in bench/accuracy.py (GraphSAGE 3x256, [15,10,5], batch 1024, trimmed,
float(loss) every step). Reports the median ms per step for each variant.

Usage: python bench/overlap_ab.py [--graph products] [--block 20] [--rounds 3]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gnnsample.graph import load
from gnnsample.loader import GpuSampler, PrefetchingGpuSampler
from gnnsample.models import SAGE, layer_adjs

FANOUTS = [15, 10, 5]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="products")
    p.add_argument("--block", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    device = torch.device("xpu")

    g = load(args.graph)
    if "feat" in g.extras:
        x = np.asarray(g.extras["feat"])
        y = torch.from_numpy(np.asarray(g.extras["label"])).to(device)
    else:
        x = np.random.default_rng(0).standard_normal((g.num_nodes, 64), dtype=np.float32)
        y = torch.randint(0, 16, (g.num_nodes,), device=device)
    seeds = np.random.default_rng(0).permutation(np.asarray(g.train_seeds()))
    n_batches = 2 * args.rounds * args.block + 4
    batches = [seeds[(i * 1024) % (len(seeds) - 1024):][:1024] for i in range(n_batches)]

    serial = GpuSampler(g.indptr, g.indices, FANOUTS, 1024, feat=x)
    pre = PrefetchingGpuSampler(g.indptr, g.indices, FANOUTS, 1024, feat=x)
    torch.manual_seed(0)
    model = SAGE(x.shape[1], 256, int(y.max()) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step(node, ei, xb, n_nodes, n_edges, bs):
        adjs = layer_adjs(ei[0], ei[1], n_nodes, n_edges, trim=True)
        opt.zero_grad()
        loss = F.cross_entropy(model(xb, adjs)[:bs], y[node[:bs]])
        loss.backward()
        opt.step()
        return float(loss)  # per-step sync, as in a training loop that logs the loss

    def run_serial(bs):
        t = []
        for i, b in enumerate(bs):
            t0 = time.perf_counter()
            step(*serial.sample(b, seed=i), len(b))
            t.append(time.perf_counter() - t0)
        return t

    def run_overlap(bs):
        t, t0 = [], time.perf_counter()
        for res in pre.iterate(bs):
            step(*res, 1024)
            now = time.perf_counter()
            t.append(now - t0)
            t0 = now
        return t

    run_serial(batches[:2])  # warm-up both paths (JIT, allocator)
    run_overlap(batches[2:4])
    times = {"serial": [], "overlap": []}
    pos = 4
    for r in range(args.rounds):
        order = [("serial", run_serial), ("overlap", run_overlap)]
        if r % 2:
            order.reverse()
        for name, fn in order:
            t = fn(batches[pos:pos + args.block])
            times[name] += t[1:]  # drop each block's first step (pipeline fill)
            pos += args.block
            print(f"round {r} {name:8} median {np.median(t[1:]) * 1e3:6.1f} ms/step", flush=True)

    s, o = np.median(times["serial"]), np.median(times["overlap"])
    print(f"{args.graph}: serial {s * 1e3:.1f} ms/step, overlap {o * 1e3:.1f} ms/step, "
          f"ratio {o / s:.3f}")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "overlap_ab.jsonl", "a") as f:
        f.write(json.dumps(dict(graph=args.graph, serial_ms=s * 1e3, overlap_ms=o * 1e3,
                                ratio=o / s, n=len(times["serial"]))) + "\n")


if __name__ == "__main__":
    main()
