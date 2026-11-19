"""D25: device memory of the batched sampler (K=8), D24 configuration vs the lean one.

  old  own device copy of the graph, direct-mapped dedup tables, worst-case slots
       (table="direct", slots="max"): what BatchedGpuSampler allocated before D25
  new  reads today's sampler's device graph, table="auto", slots sized to need ("grow")

Both sit next to today's single-minibatch Sampler, which holds the graph (as in training and
bench/e2e_batched_ab.py); its copy is the same for both and listed separately. Each
configuration is built in turn, runs --batches minibatches of 1024 seeds through
iterate_full(overlap=True) so that its slots are used (and, for "new", sized), and is freed
before the next. Bytes are exact allocation sizes: MultiSampler.memory() and the slot
tensors. torch.xpu.memory_allocated() deltas are printed as a cross-check of the slots.

Feature slots: products uses its features (shared with today's sampler, host USM). Graphs
without features use none unless --random-feat (host memory nodes x dim x 4 bytes; Twitter
at dim 100 is 16.7 GB); without them the feature-slot row is computed as 2 x slot rows x
--feat-dim x 4 bytes (slot rows = the graph slots' node capacity) and marked "(computed)".

The old configuration needs a second graph copy plus K direct tables (Twitter: ~6.7 GB +
2.7 GB), which may not fit; --configs new measures the lean one alone.

Usage: python bench/batched_memory.py --graph products [--K 8] [--batches 64]
       [--configs old,new] [--feat-dim 100] [--random-feat]
Appends one JSON line to --json-out (default results/batched_memory.jsonl).
"""

import argparse
import gc
import json
import types
from pathlib import Path

import numpy as np
import torch

from gnnsample import _C
from gnnsample.graph import DATA, load
from gnnsample.loader import BatchedGpuSampler

FANOUTS = [15, 10, 5]
BATCH = 1024
ROOT = Path(__file__).resolve().parents[1]
PARTS = ("graph", "tables", "buffers", "graph_slots", "feature_slots", "total")
LABELS = dict(graph="graph copy (MultiSampler's own)", tables="dedup tables",
              buffers="per-minibatch buffers", graph_slots="graph slots",
              feature_slots="feature slots", total="total")


def today_sampler(name, feat):
    """Today's Sampler, loaded straight from the CSC files (no host copy, D19), wrapped with
    the attributes BatchedGpuSampler's share_graph_with / share_features_with read."""
    d = DATA / name
    s = _C.Sampler.from_files(str(d / "indptr.int64.bin"), str(d / "indices.int32.bin"),
                              FANOUTS, BATCH, "gpu")
    if feat is not None:
        s.set_features(feat, "host")
    return types.SimpleNamespace(s=s, device=torch.device("xpu"), reordered=False,
                                 dim=None if feat is None else feat.shape[1])


def mb(b):
    return f"{b / 2**20:10.1f}"


def measure(cfg, g, today, batches, K, feat_dim):
    gc.collect()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    base = torch.xpu.memory_allocated()
    feats = dict(share_features_with=today) if today.dim is not None else {}
    if cfg == "old":
        bs = BatchedGpuSampler(g.indptr, g.indices, FANOUTS, BATCH, K=K, overlap=True,
                               table="direct", slots="max", **feats)
    else:
        bs = BatchedGpuSampler(None, None, FANOUTS, BATCH, K=K, overlap=True,
                               share_graph_with=today, **feats)
    at_build = bs.memory()
    nodes, edges, it = [], [], None
    for it in bs.iterate_full(batches):
        nodes.append(int(it["node"].numel()))
        edges.append(int(it["edge_index"].shape[1]))
    torch.xpu.synchronize()
    mem = bs.memory()
    torch_delta = torch.xpu.memory_allocated() - base
    computed = False
    if today.dim is None and feat_dim:  # no features: what 2 feature slots would take
        rows = max(s["node"].numel() for gs in bs.gslots for s in gs if s["node"] is not None)
        mem["feature_slots"] = 2 * rows * feat_dim * 4
        mem["total"] += mem["feature_slots"]
        computed = True
    out = dict(config=cfg, bytes={k: int(mem[k]) for k in PARTS}, table=mem["table"],
               at_build_bytes={k: int(at_build[k]) for k in PARTS},
               torch_allocated_delta=int(torch_delta), feature_slots_computed=computed,
               regrown=bs.regrown, max_nodes=bs.ms.max_nodes, max_edges=bs.ms.max_edges,
               seen_nodes=dict(max=max(nodes), mean=float(np.mean(nodes))),
               seen_edges=dict(max=max(edges), mean=float(np.mean(edges))))
    del bs, it
    gc.collect()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="products")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--batches", type=int, default=64, help="minibatches run per configuration")
    ap.add_argument("--configs", default="old,new")
    ap.add_argument("--feat-dim", type=int, default=100, help="for graphs without features")
    ap.add_argument("--random-feat", action="store_true", help="allocate random features")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    g = load(args.graph, mmap=True)
    feat = None
    if "feat" in g.extras:
        feat = np.ascontiguousarray(g.extras["feat"], dtype=np.float32)
    elif args.random_feat:
        feat = np.random.default_rng(0).standard_normal((g.num_nodes, args.feat_dim),
                                                        dtype=np.float32)
    feat_dim = feat.shape[1] if feat is not None else args.feat_dim
    today = today_sampler(args.graph, feat)
    del feat
    gc.collect()
    seeds = np.random.default_rng(0).permutation(np.asarray(g.train_seeds()))
    nb = min(args.batches, len(seeds) // BATCH)
    batches = [seeds[i * BATCH:(i + 1) * BATCH] for i in range(nb)]
    today_graph = g.indptr.nbytes + g.indices.nbytes

    res = {}
    for cfg in [c for c in args.configs.split(",") if c]:
        res[cfg] = measure(cfg, g, today, batches, args.K, feat_dim)
        r, b = res[cfg], res[cfg]["bytes"]
        slots = b["graph_slots"] + (0 if r["feature_slots_computed"] else b["feature_slots"])
        print(f"  {cfg}: table {r['table']}, slot regrowths {r['regrown']}, torch delta "
              f"{mb(r['torch_allocated_delta']).strip()} MB vs slots {mb(slots).strip()} MB; "
              f"minibatch nodes max {r['seen_nodes']['max']} (worst {r['max_nodes']}), "
              f"edges max {r['seen_edges']['max']} (worst {r['max_edges']})", flush=True)

    cfgs = list(res)
    print(f"\n{args.graph}: {g.num_nodes} nodes, K={args.K}, batch {BATCH}, {nb} minibatches, "
          f"feature dim {feat_dim}; MB (2^20 bytes)")
    print(f"{'part':34}" + "".join(f"{c:>11}" for c in cfgs) +
          (f"{'saved':>11}" if len(cfgs) == 2 else ""))
    for k in PARTS:
        label = LABELS[k] + (" (computed)" if k == "feature_slots" and
                             any(r["feature_slots_computed"] for r in res.values()) else "")
        vals = [res[c]["bytes"][k] for c in cfgs]
        print(f"{label:34}" + "".join(f" {mb(v)}" for v in vals) +
              (f" {mb(vals[0] - vals[1])}" if len(cfgs) == 2 else ""))
    print(f"{'(today sampler graph, both)':34} {mb(today_graph)}")
    dest = Path(args.json_out) if args.json_out else ROOT / "results/batched_memory.jsonl"
    with open(dest, "a") as f:
        f.write(json.dumps(dict(graph=args.graph, K=args.K, batch=BATCH, minibatches=nb,
                                feat_dim=feat_dim, today_graph_bytes=int(today_graph),
                                configs=res)) + "\n")


if __name__ == "__main__":
    main()
