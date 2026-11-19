"""A plain PyG GraphSAGE training loop on ogbn-products. The only difference between the
PyG and gnnsample runs is which NeighborLoader is imported.

    python examples/train_products.py --loader gnnsample   # SYCL sampler + GPU gather
    python examples/train_products.py --loader pyg         # torch_geometric's loader

Trains for --steps minibatches and prints the loss and the mean time per step.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models import GraphSAGE

from gnnsample.graph import load

p = argparse.ArgumentParser()
p.add_argument("--loader", choices=["gnnsample", "pyg"], default="gnnsample")
p.add_argument("--steps", type=int, default=60)
args = p.parse_args()

if args.loader == "gnnsample":
    from gnnsample import NeighborLoader
else:
    from torch_geometric.loader import NeighborLoader

# ogbn-products from data/prepare.py; edge_index = (neighbour, target), as PyG expects.
g = load("products")
dst = torch.from_numpy(np.repeat(np.arange(g.num_nodes), np.diff(g.indptr)))
data = Data(x=torch.from_numpy(np.asarray(g.extras["feat"])),
            y=torch.from_numpy(np.asarray(g.extras["label"])),
            edge_index=torch.stack([torch.from_numpy(g.indices.astype(np.int64)), dst]))
train_idx = torch.from_numpy(np.asarray(g.extras["train_idx"]))

loader = NeighborLoader(data, num_neighbors=[15, 10, 5], input_nodes=train_idx,
                        batch_size=1024, shuffle=True, num_workers=2 if args.loader == "pyg" else 0)

device = torch.device("xpu")
model = GraphSAGE(data.num_features, 256, num_layers=3, out_channels=int(data.y.max()) + 1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

times, t0 = [], time.perf_counter()
for step, batch in enumerate(loader):
    batch = batch.to(device)  # no-op for gnnsample: tensors are already on the XPU
    opt.zero_grad()
    out = model(batch.x, batch.edge_index,
                num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                num_sampled_edges_per_hop=batch.num_sampled_edges)[: batch.batch_size]
    loss = F.cross_entropy(out, batch.y[: batch.batch_size])
    loss.backward()
    opt.step()
    torch.xpu.synchronize()
    now = time.perf_counter()
    if step >= 5:  # skip warm-up (kernel compilation, worker start)
        times.append(now - t0)
    t0 = now
    if step % 20 == 0:
        print(f"step {step:3d}  loss {float(loss):.4f}", flush=True)
    if step + 1 >= args.steps:
        break
print(f"{args.loader}: {np.mean(times) * 1e3:.0f} ms per training step "
      f"(mean of steps 6-{args.steps})")
