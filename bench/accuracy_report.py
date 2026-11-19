"""Aggregate results/accuracy/*.json: per-sampler mean ± std over seeds, the paired
per-seed difference, and Welch's t-test for "same accuracy" (parity criterion: the
difference in means is within seed-to-seed noise)."""

import json
from pathlib import Path

import numpy as np
from scipy import stats

DIR = Path(__file__).resolve().parents[1] / "results" / "accuracy"


def main():
    runs = {}
    for f in sorted(DIR.glob("*_seed*.json")):
        if f.stem.endswith("_smoke"):
            continue
        r = json.loads(f.read_text())
        runs.setdefault(r["sampler"], {})[r["seed"]] = r
    samplers = sorted(runs)
    print(f"{'sampler':8} {'seeds':>5} {'val (last epoch)':>20} {'test 50K':>18} "
          f"{'train s/epoch':>14}")
    vals, tests = {}, {}
    for s in samplers:
        rs = [runs[s][k] for k in sorted(runs[s])]
        vals[s] = np.array([r["epochs"][-1]["val_acc"] for r in rs])
        tests[s] = np.array([r["test_acc_50k"] for r in rs])
        tps = np.mean([np.mean([e["train_s"] for e in r["epochs"][1:]]) for r in rs])
        print(f"{s:8} {len(rs):5} {vals[s].mean():.4f} ± {vals[s].std(ddof=1) if len(rs) > 1 else 0:.4f}"
              f"     {tests[s].mean():.4f} ± {tests[s].std(ddof=1) if len(rs) > 1 else 0:.4f}"
              f"   {tps:12.1f}")
    if len(samplers) == 2 and all(len(vals[s]) > 1 for s in samplers):
        a, b = samplers
        for name, d in (("val", vals), ("test", tests)):
            t, p = stats.ttest_ind(d[a], d[b], equal_var=False)
            print(f"{name}: mean diff {a}-{b} = {d[a].mean() - d[b].mean():+.4f}, "
                  f"Welch t = {t:.2f}, p = {p:.3f}")
        print("per-epoch val accuracy (mean over seeds):")
        n_ep = min(len(r["epochs"]) for s in samplers for r in runs[s].values())
        for ep in range(n_ep):
            row = "  ".join(f"{s} {np.mean([runs[s][k]['epochs'][ep]['val_acc'] for k in runs[s]]):.4f}"
                            for s in samplers)
            print(f"  epoch {ep}: {row}")


if __name__ == "__main__":
    main()
