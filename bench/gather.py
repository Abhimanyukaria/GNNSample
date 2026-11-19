"""Feature-gather micro-benchmark (follow-up to D9 measurement 2, where the CPU gather was
47% of the step at only ~0.45 GB/s).

Gathers the feature rows of real sampled minibatches (ogbn-products, [15,10,5], batch 1024;
~409 K rows × 100 floats = 164 MB per minibatch) with:
  cpu-index_select   torch.index_select (what PyG's loader does), thread sweep
  cpu-sorted         same with the index sorted (how much is locality?)
  cpu+copy           best CPU gather + copy of the result to the iGPU (today's pipeline)
  torch-xpu          table resident on the iGPU, torch index_select there
  sycl-<usm>         our SYCL kernel, table in device / host / shared USM
Every GPU variant leaves the result on the device, ready for the model; the CPU-only
variants still need the copy (measured by cpu+copy).
Bandwidth = bytes read + written / time. "contiguous" rows give each variant's ceiling.

Usage: python bench/gather.py [--batches 10] [--reps 5]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pyg_lib.sampler import neighbor_sample

from gnnsample import _C
from gnnsample.graph import load

FANOUTS = [15, 10, 5]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--threads", default="1,2,4,7,14")
    args = p.parse_args()

    g = load("products")
    x = torch.from_numpy(np.ascontiguousarray(g.extras["feat"], dtype=np.float32))
    rows_total, dim = x.shape
    torch.manual_seed(0)
    rowptr = torch.from_numpy(g.indptr)
    col = torch.from_numpy(g.indices.astype(np.int64))
    train = torch.from_numpy(np.asarray(g.extras["train_idx"]))
    perm = train[torch.randperm(len(train))]
    idx = [neighbor_sample(rowptr, col, perm[b * 1024:(b + 1) * 1024], FANOUTS, csc=True)[2]
           for b in range(args.batches)]
    del rowptr, col
    n_rows = int(np.mean([len(i) for i in idx]))
    contiguous = [torch.arange(n_rows)]
    print(f"table {rows_total:,} x {dim} ({x.numel() * 4 / 1e9:.2f} GB), "
          f"{args.batches} batches, mean {n_rows:,} rows ({n_rows * dim * 4 / 1e6:.0f} MB)",
          flush=True)

    rows = []

    def record(name, config, per_batch_times, index_sets):
        t = float(np.median(per_batch_times))
        n = np.mean([len(i) for i in index_sets])
        gbs = 2 * n * dim * 4 / t / 1e9
        rows.append(dict(variant=name, config=config, median_s=t, gb_per_s=gbs))
        print(f"{name:18} {config:22} {t * 1e3:8.1f} ms {gbs:7.2f} GB/s", flush=True)

    def per_batch(fn, index_sets):
        times = []
        for i in index_sets:
            fn(i)  # warm-up for this index
            times.append(np.median(timed(lambda: fn(i), args.reps)))
        return times

    # CPU gathers
    out = torch.empty(max(len(i) for i in idx), dim)
    for t in map(int, args.threads.split(",")):
        torch.set_num_threads(t)
        f = lambda i: torch.index_select(x, 0, i, out=out[: len(i)])
        record("cpu-index_select", f"threads={t}", per_batch(f, idx), idx)
    best_t = int(args.threads.split(",")[-1])
    torch.set_num_threads(best_t)
    f = lambda i: torch.index_select(x, 0, i, out=out[: len(i)])
    record("cpu-index_select", "contiguous", per_batch(f, contiguous), contiguous)
    sorted_idx = [torch.sort(i).values for i in idx]
    record("cpu-sorted", f"threads={best_t}", per_batch(f, sorted_idx), sorted_idx)

    def cpu_then_copy(i):
        torch.index_select(x, 0, i, out=out[: len(i)])
        out[: len(i)].to("xpu")
        torch.xpu.synchronize()

    record("cpu+copy", f"threads={best_t}", per_batch(cpu_then_copy, idx), idx)

    # Table resident on the iGPU, PyTorch's own gather
    x_xpu = x.to("xpu")
    idx_xpu = [i.to("xpu") for i in idx]

    def torch_xpu(i):
        x_xpu.index_select(0, i)
        torch.xpu.synchronize()

    record("torch-xpu", "table on device", per_batch(torch_xpu, idx_xpu), idx)
    del x_xpu, idx_xpu
    torch.xpu.empty_cache()

    # Our SYCL kernel; the index is copied in beforehand, so only the gather is timed
    check = torch.index_select(x, 0, idx[0]).numpy()
    for usm in ("device", "host", "shared"):
        gk = _C.Gather(x.numpy(), usm, "gpu")
        assert np.array_equal(gk.gather(idx[0].numpy().astype(np.int32)), check)
        for name, sets in (("sycl-" + usm, idx), ("sycl-" + usm, contiguous)):
            times = [float(np.median(gk.time(i.numpy().astype(np.int32), args.reps)[1:]))
                     for i in sets]
            record(name, "contiguous" if sets is contiguous else "sampled rows", times, sets)
        del gk

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gather_products.json").write_text(json.dumps(dict(
        rows_total=rows_total, dim=dim, batches=args.batches, mean_rows=n_rows,
        torch=torch.__version__, results=rows), indent=2))


if __name__ == "__main__":
    main()
