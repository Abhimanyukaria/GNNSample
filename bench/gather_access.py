"""Why is the CPU gather 23 ms when an index is repeated but ~360 ms per fresh minibatch?

Times torch.index_select of ~410 K feature rows (products) for:
  same-index      one sampled index, repeated (what bench/gather.py measured)
  fresh-sampled   a new pyg-lib minibatch index every call (what training does)
  fresh-random    uniformly random rows every call
  fresh-sorted    fresh sampled index, sorted
and, for the fresh sampled indices, the same gather on the iGPU (SYCL, table in host USM)
plus host->device copies (pageable and pinned) of a 164 MB block.
"""

import time

import numpy as np
import torch
from pyg_lib.sampler import neighbor_sample

from gnnsample import _C
from gnnsample.graph import load


def run(label, fn, idx_list):
    ts = []
    for i in idx_list:
        t0 = time.perf_counter()
        fn(i)
        ts.append(time.perf_counter() - t0)
    print(f"{label:46} {np.median(ts[1:]) * 1e3:8.1f} ms  (first {ts[0] * 1e3:.0f} ms)",
          flush=True)


def main():
    g = load("products")
    x = torch.from_numpy(np.ascontiguousarray(g.extras["feat"], dtype=np.float32))
    torch.set_num_threads(7)
    torch.manual_seed(0)
    rowptr = torch.from_numpy(g.indptr)
    col = torch.from_numpy(g.indices.astype(np.int64))
    train = torch.from_numpy(np.asarray(g.extras["train_idx"]))
    perm = train[torch.randperm(len(train))]
    fresh = [neighbor_sample(rowptr, col, perm[b * 1024:(b + 1) * 1024], [15, 10, 5],
                             csc=True)[2] for b in range(12)]
    n = len(fresh[0])
    rng = np.random.default_rng(0)
    rand = [torch.from_numpy(rng.choice(x.shape[0], n, replace=False)) for _ in range(12)]
    out = torch.empty(max(map(len, fresh)), x.shape[1])
    sel = lambda i: torch.index_select(x, 0, i, out=out[: len(i)])

    print(f"rows per batch ~{n:,}; unique 4 KB pages touched by one sampled batch: "
          f"{len(np.unique(fresh[0].numpy() * 400 // 4096)):,} of {x.numel() * 4 // 4096:,}")
    run("cpu same-index (repeated 12x)", sel, [fresh[0]] * 12)
    run("cpu fresh-sampled", sel, fresh)
    run("cpu fresh-random", sel, rand)
    run("cpu fresh-sorted", sel, [torch.sort(i).values for i in fresh])

    sync = torch.xpu.synchronize
    block = out[:n].contiguous()
    run("copy 164 MB pageable -> xpu", lambda _: (block.to("xpu"), sync()), fresh)
    pinned = block.pin_memory()
    run("copy 164 MB pinned -> xpu", lambda _: (pinned.to("xpu", non_blocking=True), sync()),
        fresh)

    gk = _C.Gather(x.numpy(), "host", "gpu")
    fresh32 = [i.numpy().astype(np.int32) for i in fresh]
    t = [gk.time(i, 1)[0] for i in fresh32]
    print(f"{'sycl-host fresh-sampled (gather only)':46} {np.median(t[1:]) * 1e3:8.1f} ms  "
          f"(first {t[0] * 1e3:.0f} ms)")
    t = [gk.time(fresh32[0], 1)[0] for _ in range(12)]
    print(f"{'sycl-host same-index':46} {np.median(t[1:]) * 1e3:8.1f} ms")


if __name__ == "__main__":
    main()
