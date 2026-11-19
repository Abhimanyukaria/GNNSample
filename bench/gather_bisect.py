"""Bisect: the same CPU index_select is ~20 ms in bench/gather_access.py but ~330 ms inside
bench/profile_pyg.py. Time it in one process while adding, step by step, what the profile
loop does around it."""

import time

import numpy as np
import torch
import torch.nn.functional as F
from pyg_lib.sampler import neighbor_sample

from gnnsample.graph import load

torch.set_num_threads(7)
g = load("products")
x = torch.from_numpy(np.asarray(g.extras["feat"]))
y = torch.from_numpy(np.asarray(g.extras["label"]))
rowptr = torch.from_numpy(g.indptr)
col = torch.from_numpy(g.indices.astype(np.int64))
train = torch.from_numpy(np.asarray(g.extras["train_idx"]))
torch.manual_seed(0)
perm = train[torch.randperm(len(train))]
batches = [perm[b * 1024:(b + 1) * 1024] for b in range(40)]
pre = [neighbor_sample(rowptr, col, s, [15, 10, 5], csc=True) for s in batches[:8]]


def gather_ms(node):
    t0 = time.perf_counter()
    torch.index_select(x, 0, node)
    return (time.perf_counter() - t0) * 1e3


def report(label, ts):
    print(f"{label:58} {np.median(ts[1:]):8.1f} ms", flush=True)


report("A. pre-sampled indices, back to back", [gather_ms(o[2]) for o in pre])

ts = []
for s in batches[:8]:
    out = neighbor_sample(rowptr, col, s, [15, 10, 5], csc=True)
    ts.append(gather_ms(out[2]))
report("B. sample then gather, alternating", ts)

from gnnsample.models import SAGE, layer_adjs, orient  # noqa: E402  (imports torch_geometric)

report("C. after importing torch_geometric (pre-sampled)", [gather_ms(o[2]) for o in pre])

device = torch.device("xpu")
model = SAGE(x.shape[1], 256, int(y.max()) + 1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_step(out, seeds):
    row, colx, node, _, n_nodes, n_edges = out
    src, dst = orient(row, colx, n_edges, len(seeds))
    adjs = [(e.to(device), n) for e, n in layer_adjs(src, dst, n_nodes, n_edges, trim=True)]
    xb = torch.index_select(x, 0, node).to(device)
    opt.zero_grad()
    F.cross_entropy(model(xb, adjs)[: len(seeds)], y[node[: len(seeds)]].to(device)).backward()
    opt.step()
    torch.xpu.synchronize()


ts = []
for s in batches[8:20]:
    out = neighbor_sample(rowptr, col, s, [15, 10, 5], csc=True)
    ts.append(gather_ms(out[2]))
    train_step(out, s)
report("D. sample, gather, iGPU train step (profile loop)", ts)

report("E. pre-sampled again, after training has run", [gather_ms(o[2]) for o in pre])

ts = []
for s in batches[20:32]:
    out = neighbor_sample(rowptr, col, s, [15, 10, 5], csc=True)
    train_step(out, s)
    time.sleep(0.2)
    ts.append(gather_ms(out[2]))
report("F. as D but 200 ms idle before each gather", ts)
