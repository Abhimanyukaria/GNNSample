"""D24: end-to-end training A/B, today's sampler vs batched (K minibatches per launch).

One process, one feature table (shared), same minibatches and keys for every
configuration, so every configuration trains on bit-identical inputs. Each configuration
runs one full epoch per round (model re-initialised with the same seed), and rounds rotate
the configuration order so the iGPU's drift spreads over all of them.

  today-serial     GpuSampler path, no overlap: sample -> export -> gather -> train
  batched-serial   BatchedGpuSampler(K), no overlap: one launch per K minibatches
  today-overlap    PrefetchingGpuSampler (production): minibatch i+1 while training i
  batched-overlap  BatchedGpuSampler(K, overlap): next block while training this one

Reported per configuration and round:
  epoch_s       wall time of the whole epoch, from the first request to the last step's end,
                INCLUDING the start-up wait for the first minibatch (the pipeline fill)
  first_ms      that start-up wait alone (batched waits for its whole first block)
  per step      wait (consumer blocked on data), train (adjacency build + forward + backward
                + optimizer, synchronised), and producer stages sample / export / gather
                (serial configurations; batched also reports them when overlapped)
  loss check    max |loss - loss(today-serial)| over the epoch's steps: identical inputs, so
                only non-deterministic float atomics can make them differ

Graphs without features (LiveJournal, Orkut) get random float32 features (--feat-dim) and
random labels (47 classes), the gSampler convention; timing only.

Usage: python bench/e2e_batched_ab.py --graph products [--K 8] [--rounds 2]
       [--configs today-serial,batched-serial,today-overlap,batched-overlap] [--max-batches N]
Appends one JSON line to --json-out (default results/e2e_batched_ab.jsonl).
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gnnsample.graph import load
from gnnsample.loader import BatchedGpuSampler, PrefetchingGpuSampler
from gnnsample.models import SAGE, layer_adjs

FANOUTS = [15, 10, 5]
ROOT = Path(__file__).resolve().parents[1]
STAGES = ("sample", "export", "gather", "slot_wait")


def today_serial_items(gs, batches):
    """The GpuSampler path (run -> export_to -> gather), with each stage timed."""
    dev = gs.device
    for i, b in enumerate(batches):
        t0 = time.perf_counter()
        c = gs.s.run(np.ascontiguousarray(b, dtype=np.int32), i)
        t1 = time.perf_counter()
        n, e = c["num_nodes"], c["num_edges"]
        node = torch.empty(n, dtype=torch.int64, device=dev)
        ei = torch.empty(2, e, dtype=torch.int64, device=dev)
        x = torch.empty(n, gs.dim, device=dev)
        torch.xpu.current_stream().synchronize()
        gs.s.export_to(node.data_ptr(), ei.data_ptr(), 0, 0)
        t2 = time.perf_counter()
        gs.s.gather(node.data_ptr(), n, x.data_ptr())
        t3 = time.perf_counter()
        yield dict(node=node, edge_index=ei, x=x, num_sampled_nodes=c["num_sampled_nodes"],
                   num_sampled_edges=c["num_sampled_edges"],
                   t=dict(sample=t1 - t0, export=t2 - t1, gather=t3 - t2, slot_wait=0.0))


def train_epoch(items, batches, y_dev, make_model):
    model, opt = make_model()
    torch.xpu.synchronize()
    rec = dict(wait=[], train=[], loss=[], **{s: [] for s in STAGES})
    t_start = t_req = time.perf_counter()
    first = None
    for i, it in enumerate(items):
        t_got = time.perf_counter()
        if first is None:
            first = t_got - t_start
        bs = len(batches[i])
        ei = it["edge_index"]
        adjs = layer_adjs(ei[0], ei[1], it["num_sampled_nodes"], it["num_sampled_edges"], trim=True)
        opt.zero_grad()
        loss = F.cross_entropy(model(it["x"], adjs)[:bs], y_dev[it["node"][:bs]])
        loss.backward()
        opt.step()
        rec["loss"].append(loss.item())  # synchronises the step
        t_done = time.perf_counter()
        rec["wait"].append(t_got - t_req)
        rec["train"].append(t_done - t_got)
        for s in STAGES:
            rec[s].append(it.get("t", {}).get(s, float("nan")))
        t_req = t_done
    rec["epoch_s"] = t_req - t_start
    rec["first_ms"] = first * 1e3
    # On WSL2 the iGPU's memory is taken from Windows, outside the VM: hand PyTorch's cached
    # blocks back after every epoch (products was killed by the low-memory guard in round 1).
    del model, opt, loss, it, adjs
    gc.collect()
    torch.xpu.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="products")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--configs", default="today-serial,batched-serial,today-overlap,batched-overlap")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = every full minibatch")
    ap.add_argument("--feat-dim", type=int, default=100, help="random features if none")
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device("xpu")

    g = load(args.graph)
    rng = np.random.default_rng(0)
    if "feat" in g.extras:
        feat = np.ascontiguousarray(g.extras["feat"], dtype=np.float32)
        y = torch.from_numpy(np.asarray(g.extras["label"]).reshape(-1)).long()
    else:
        feat = rng.standard_normal((g.num_nodes, args.feat_dim), dtype=np.float32)
        y = torch.from_numpy(rng.integers(0, 47, g.num_nodes))
    n_classes = int(y.max()) + 1
    seeds = np.random.default_rng(0).permutation(np.asarray(g.train_seeds()))
    nb = len(seeds) // 1024
    if args.max_batches:
        nb = min(nb, args.max_batches)
    batches = [seeds[i * 1024:(i + 1) * 1024] for i in range(nb)]
    y_dev = y.to(dev)

    today = PrefetchingGpuSampler(g.indptr, g.indices, FANOUTS, 1024, feat=feat)
    dim = feat.shape[1]
    del feat
    gc.collect()
    # One instance for both batched configurations (K-minibatch buffers and slots; two did
    # not fit next to training). D25: it reads today's device graph instead of a second
    # copy (Orkut: 0.94 GB, which got the D24 run killed) and uses the lean tables / slots.
    batched = BatchedGpuSampler(None, None, FANOUTS, 1024, K=args.K, share_features_with=today,
                                share_graph_with=today)
    print(f"batched sampler device memory: {batched.memory()}", flush=True)

    def make_model():
        torch.manual_seed(0)
        model = SAGE(dim, 256, n_classes).to(dev)
        return model, torch.optim.Adam(model.parameters(), lr=1e-3)

    def items(cfg, bl):
        if cfg == "today-serial":
            return today_serial_items(today, bl)
        if cfg == "today-overlap":
            return today.iterate_full(bl)
        batched.overlap = cfg == "batched-overlap"
        return batched.iterate_full(bl)

    configs = [c for c in args.configs.split(",") if c]
    # Warm-up: JIT and first table resets for every path, and the model's kernels.
    for cfg in configs:
        warm = batches[: 2 * args.K]
        train_epoch(items(cfg, warm), warm, y_dev, make_model)

    runs = {c: [] for c in configs}
    for r in range(args.rounds):
        order = configs[r % len(configs):] + configs[: r % len(configs)]
        for cfg in order:
            rec = train_epoch(items(cfg, batches), batches, y_dev, make_model)
            runs[cfg].append(rec)
            print(f"  round {r} {cfg:16} epoch {rec['epoch_s']:7.2f} s  first {rec['first_ms']:7.1f} ms",
                  flush=True)

    # Slots are sized on first use (D25), so this is the steady state.
    print(f"batched sampler device memory after the runs: {batched.memory()} "
          f"(slot regrowths: {batched.regrown})", flush=True)
    ref = runs[configs[0]]
    print(f"\n{args.graph}: {nb} minibatches/epoch, K={args.K}, {args.rounds} rounds "
          f"(means over rounds; stage columns are per-minibatch means, ms)")
    print(f"{'config':16} {'epoch s':>8} {'vs ' + configs[0]:>18} {'first ms':>9} {'wait':>7} "
          f"{'train':>7} {'sample':>7} {'export':>7} {'gather':>7} {'max|dloss|':>11}")
    rows = []
    for cfg in configs:
        recs = runs[cfg]
        ep = np.array([x["epoch_s"] for x in recs])
        ratio = float(np.mean(ep / np.array([x["epoch_s"] for x in ref])))
        st = {s: float(np.nanmean([np.nanmean(x[s]) for x in recs]) * 1e3) for s in STAGES}
        wait = float(np.mean([np.mean(x["wait"]) for x in recs]) * 1e3)
        train = float(np.mean([np.mean(x["train"]) for x in recs]) * 1e3)
        dloss = float(max(np.max(np.abs(np.array(a["loss"]) - np.array(b["loss"])))
                          for a, b in zip(recs, ref)))
        row = dict(config=cfg, epoch_s=ep.tolist(), epoch_ratio_vs_first=ratio,
                   first_ms=[x["first_ms"] for x in recs], wait_ms=wait, train_ms=train,
                   train_median_ms=float(np.median(np.concatenate([x["train"] for x in recs])) * 1e3),
                   stage_ms=st, max_abs_loss_diff=dloss,
                   final_loss=[x["loss"][-1] for x in recs])
        rows.append(row)
        print(f"{cfg:16} {ep.mean():8.2f} {ratio:18.3f} {np.mean(row['first_ms']):9.1f} {wait:7.1f} "
              f"{train:7.1f} {st['sample']:7.1f} {st['export']:7.1f} {st['gather']:7.1f} {dloss:11.2e}")
    dest = Path(args.json_out) if args.json_out else ROOT / "results/e2e_batched_ab.jsonl"
    with open(dest, "a") as f:
        f.write(json.dumps(dict(graph=args.graph, K=args.K, rounds=args.rounds, minibatches=nb,
                                rows=rows)) + "\n")


if __name__ == "__main__":
    main()
