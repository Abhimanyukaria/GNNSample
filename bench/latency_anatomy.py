"""Where does one minibatch's time go? (sizing the latency problems by order of magnitude)

For each kernel stage of the no-sync sampler (final options), from GPU timestamps on a
profiling queue (profile mode 3, no extra barriers):
  busy   GPU execution time of the stage's last kernel
  gap    idle time on the GPU before that kernel started (launch + kernel-boundary drain:
         the cost of synchronising the whole GPU between phases)
plus host time outside the GPU timeline (Python, seed copy, final readback, waits).
Scan stages contain several kernels; their busy time is reported as the span from the
previous stage's end, so their internal gaps are included in it.

Usage: python bench/latency_anatomy.py [--graphs products,orkut] [--batch 1024] [--batches 20]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gnnsample import _C
from gnnsample.graph import load
from gnnsample.loader import DEFAULT_OPTIONS

FANOUTS = [15, 10, 5]
MULTI = {"scan-count", "scan-flag"}  # stages made of several kernels


def anatomy(name, batch, n):
    g = load(name, extras=False, mmap=True)
    d = Path(g.indices.filename).parent
    s = _C.Sampler.from_files(str(d / "indptr.int64.bin"), str(d / "indices.int32.bin"),
                              FANOUTS, batch, "gpu", True)
    s.set_options(DEFAULT_OPTIONS)
    rng = np.random.default_rng(0)
    batches = [rng.choice(g.num_nodes, batch, replace=False).astype(np.int32) for _ in range(n + 3)]
    for i in range(3):
        s.run(batches[i], i)
    s.set_profile(3)
    per = defaultdict(lambda: defaultdict(list))  # (hop, stage) -> busy/gap lists
    totals = defaultdict(list)
    for i, b in enumerate(batches[3:]):
        s.run(b, i)
        tr = s.trace()
        hop, prev_end = -1, None
        busy_sum = gap_sum = 0.0
        for stage, start, end in tr:
            if stage == "count":
                hop += 1
            key = (max(hop, 0) if stage not in ("reset", "seeds") else "setup", stage)
            if prev_end is None or end < prev_end:  # first entry, or a stage with no kernel
                prev_end = end
                continue
            if stage in MULTI:
                busy, gap = end - prev_end, 0.0
            else:
                gap = max(0.0, start - prev_end)
                busy = end - start
            per[key]["busy"].append(busy / 1e3)
            per[key]["gap"].append(gap / 1e3)
            busy_sum += busy
            gap_sum += gap
            prev_end = end
        gpu_span = tr[-1][2] - min(t[1] for t in tr if t[2] >= tr[0][2])
        totals["host_us"].append(s.trace_host_ns() / 1e3)
        totals["gpu_busy_us"].append(busy_sum / 1e3)
        totals["gpu_gap_us"].append(gap_sum / 1e3)
        totals["gpu_span_us"].append(gpu_span / 1e3)
    med = {k: float(np.median(v)) for k, v in totals.items()}
    med["host_outside_gpu_us"] = med["host_us"] - med["gpu_span_us"]
    stages = {f"{k[0]}:{k[1]}": {m: float(np.median(v)) for m, v in d2.items()}
              for k, d2 in per.items()}
    return dict(graph=name, batch=batch, batches=n, totals_us=med, stages_us=stages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="products,orkut")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    for name in a.graphs.split(","):
        r = anatomy(name, a.batch, a.batches)
        t = r["totals_us"]
        print(f"\n{name}, batch {a.batch}: host wall {t['host_us'] / 1e3:.2f} ms = GPU span "
              f"{t['gpu_span_us'] / 1e3:.2f} ms + host-only {t['host_outside_gpu_us'] / 1e3:.2f} ms; "
              f"GPU busy {t['gpu_busy_us'] / 1e3:.2f} ms, GPU idle gaps {t['gpu_gap_us'] / 1e3:.2f} ms")
        print(f"  {'stage':22} {'busy us':>9} {'gap us':>8}")
        for k, v in r["stages_us"].items():
            print(f"  {k:22} {v['busy']:9.1f} {v['gap']:8.1f}")
        if a.json_out:
            with open(a.json_out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
