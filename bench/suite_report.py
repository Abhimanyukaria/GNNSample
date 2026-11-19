"""Write results/suite/iteration_<x>.md (and summary.json) for one benchmark-suite run.

Reads the files tools/bench_suite.sh left in results/suite/iter_<x>/, collects versions and
device info from this container, and compares the headline numbers with the previous
iteration's summary.json when there is one.

Usage: python bench/suite_report.py --iter 3
"""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "results" / "suite"


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt(v, d=1, suffix=""):
    return "–" if v is None else f"{v:.{d}f}{suffix}"


def delta(cur, prev, lower_is_better=True):
    if cur is None or prev is None or prev == 0:
        return ""
    change = (cur - prev) / prev * 100
    better = change < 0 if lower_is_better else change > 0
    return f" ({change:+.1f}% {'better' if better else 'worse'})" if abs(change) >= 0.05 else " (=)"


def versions():
    v = {"python": platform.python_version()}
    for mod in ("torch", "torch_geometric", "pyg_lib", "numpy"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            v[mod] = "not installed"
    try:
        out = subprocess.run(["icpx", "--version"], capture_output=True, text=True, timeout=30)
        v["icpx"] = out.stdout.splitlines()[0]
    except Exception:  # noqa: BLE001
        v["icpx"] = "unknown"
    try:
        out = subprocess.run([str(ROOT / "build" / "device_info")], capture_output=True,
                             text=True, timeout=60)
        v["devices"] = [line.strip() for line in out.stdout.splitlines() if " | " in line
                        and not line.startswith(" ")]
    except Exception:  # noqa: BLE001
        v["devices"] = []
    return v


def accuracy_summary(d):
    runs = {}
    for f in sorted(d.glob("*_seed*.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["sampler"], []).append(r)
    if not runs:
        return None
    out = {}
    for s, rs in runs.items():
        val = np.array([r["epochs"][-1]["val_acc"] for r in rs])
        test = np.array([r["test_acc_50k"] for r in rs])
        ep = np.mean([np.mean([e["train_s"] for e in r["epochs"][1:]]) for r in rs])
        out[s] = dict(seeds=len(rs), val=float(val.mean()), val_sd=float(val.std(ddof=1)) if len(rs) > 1 else 0.0,
                      test=float(test.mean()), test_sd=float(test.std(ddof=1)) if len(rs) > 1 else 0.0,
                      epoch_s=float(ep), test_all=test.tolist(), val_all=val.tolist())
    if "ours" in out and "pyg" in out and out["ours"]["seeds"] > 1 and out["pyg"]["seeds"] > 1:
        from scipy import stats

        out["welch_p_test"] = float(stats.ttest_ind(out["ours"]["test_all"], out["pyg"]["test_all"],
                                                    equal_var=False).pvalue)
        out["welch_p_val"] = float(stats.ttest_ind(out["ours"]["val_all"], out["pyg"]["val_all"],
                                                   equal_var=False).pvalue)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    args = ap.parse_args()
    x = args.iter
    d = SUITE / f"iter_{x}"
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    prev_path = SUITE / f"iter_{x - 1}" / "summary.json"
    prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    pv = prev.get("datasets", {})

    steps = []
    if (d / "steps.tsv").exists():
        for line in (d / "steps.tsv").read_text().splitlines():
            name, rc, secs = line.split("\t")
            steps.append((name, int(rc), int(secs)))

    tests = (d / "tests.txt").read_text().strip() if (d / "tests.txt").exists() else "not run"
    summary = dict(iteration=x, meta=meta, tests=tests, datasets={}, e2e=None, accuracy=None,
                   twitter=None)
    datasets = meta.get("datasets", [])

    # ---------------- per-dataset numbers
    for g in datasets:
        ds = {}
        ab = jsonl(d / f"ablate_{g}.jsonl")
        if ab:
            r = ab[-1]
            final = next((v for v in r["variants"] if v["options"] != 0), None)
            v1 = next((v for v in r["variants"] if v["options"] == 0), None)
            ds.update(gpu_final_ms=final["median_ms"] if final else None,
                      gpu_v1_ms=v1["median_ms"] if v1 else None,
                      pyg_ms=r["pyg_median_ms"],
                      speedup_vs_pyg=final["speedup_vs_pyg"] if final else None,
                      # the final options are the base, so v1's ratio is t_v1 / t_final
                      ratio_final_vs_v1=v1["paired_ratio_vs_base"] if v1 else None,
                      options=final["options"] if final else None)
        st = jsonl(d / f"stages_{g}.jsonl")
        if st:
            ds["stages_ms"] = st[-1].get("stages_ms")
            ds["pipeline_ms"] = st[-1].get("sycl_pipeline_ms")
        cpu = jsonl(d / f"cpu_{g}.jsonl")
        if cpu:
            table = {}
            for r in cpu:
                table.setdefault(r["threads"], {}).setdefault(r["sampler"], []).append(r["batches_per_s"])
            ds["cpu"] = {t: {s: float(np.median(v)) for s, v in row.items()} for t, row in sorted(table.items())}
        summary["datasets"][g] = ds

    e2e = jsonl(d / "e2e.jsonl")
    if e2e:
        summary["e2e"] = {r["rows"][0]["config"]: r["rows"][0] for r in e2e}
    acc = accuracy_summary(d / "accuracy") if (d / "accuracy").exists() else None
    summary["accuracy"] = acc
    tw = jsonl(d / "scale_twitter.jsonl")
    if tw:
        summary["twitter"] = {r["mode"]: r for r in tw}
    summary["versions"] = versions()

    # ---------------- markdown
    L = []
    add = L.append
    add(f"# gnnsample benchmark suite: iteration {x}\n")
    add(f"- **Run:** {meta.get('start', '?')} to {meta.get('end', '?')} UTC "
        f"({meta.get('minutes', '?')} min)")
    add(f"- **Code:** commit `{meta.get('commit', '?')}`"
        f"{' with uncommitted changes' if meta.get('dirty') else ''}; sampler options "
        f"{next((ds.get('options') for ds in summary['datasets'].values() if ds.get('options')), '?')}")
    add(f"- **Power:** {meta.get('power_start', '?')} at start, {meta.get('power_end', '?')} at end. "
        "Speedups below are measured within one run, so both sides see the same power state.")
    sv = summary["versions"]
    add(f"- **Software:** torch {sv.get('torch')}, torch_geometric {sv.get('torch_geometric')}, "
        f"pyg_lib {sv.get('pyg_lib')}, {sv.get('icpx')}")
    if sv.get("devices"):
        add(f"- **Devices:** {'; '.join(sv['devices'])}")
    add(f"- **Compared with:** {'iteration ' + str(x - 1) if prev else 'nothing (first iteration)'}\n")

    add("## Steps\n")
    add("| step | result | minutes |\n|---|---|---|")
    for name, rc, secs in steps:
        log = ROOT / "results" / "logs" / f"suite{x}-{name}.log"
        text = log.read_text(errors="ignore") if log.exists() else ""
        if rc == 0:
            status = "ok"
        elif rc == 137 and ("median" in text or "passed" in text):
            # The watcher's low-memory guard (or an OOM kill) hit after the result was written.
            status = "ok, result written; **killed at shutdown (exit 137, low memory)**"
        elif rc == 137:
            status = "**killed (exit 137): low-memory guard or out of memory**"
        elif rc == 124:
            status = "**timed out**"
        else:
            status = f"**failed (exit {rc})**"
        add(f"| {name} | {status} | {secs / 60:.1f} |")
    add("")
    add("## Correctness\n")
    add(f"`pytest`: {tests}\n")

    if summary["datasets"]:
        add("## GPU sampler vs pyg-lib and v1\n")
        add("Milliseconds per minibatch (batch 1024, fanout [15, 10, 5]), median over the same "
            "batches. Ours and v1 are interleaved batch by batch; pyg-lib is one process, one call "
            "per batch.\n")
        add("| dataset | ours (ms) | v1 (ms) | pyg-lib (ms) | vs pyg-lib | vs v1 (paired) |")
        add("|---|---|---|---|---|---|")
        for g, ds in summary["datasets"].items():
            if "gpu_final_ms" not in ds:
                continue
            p = pv.get(g, {})
            add(f"| {g} | {fmt(ds['gpu_final_ms'])}{delta(ds['gpu_final_ms'], p.get('gpu_final_ms'))} "
                f"| {fmt(ds['gpu_v1_ms'])} | {fmt(ds['pyg_ms'])} "
                f"| **{fmt(ds['speedup_vs_pyg'], 2, '×')}**{delta(ds['speedup_vs_pyg'], p.get('speedup_vs_pyg'), False)} "
                f"| {fmt(ds['ratio_final_vs_v1'], 2, '×')} |")
        add("")
        stage_rows = [(g, ds["stages_ms"]) for g, ds in summary["datasets"].items() if ds.get("stages_ms")]
        if stage_rows:
            names = list(stage_rows[0][1])
            add("### GPU time per stage (ms, GPU timestamps)\n")
            add("| dataset | " + " | ".join(names) + " | total |")
            add("|---|" + "---|" * (len(names) + 1))
            for g, s in stage_rows:
                add(f"| {g} | " + " | ".join(fmt(s.get(n), 2) for n in names)
                    + f" | {fmt(sum(s.values()), 1)} |")
            add("")

        cpu_rows = [(g, ds["cpu"]) for g, ds in summary["datasets"].items() if ds.get("cpu")]
        if cpu_rows:
            add("## CPU sampler vs pyg-lib at equal thread counts\n")
            add("Minibatches per second (sampling only). Ours: the SYCL sampler on the CPU device, "
                "one minibatch at a time, its runtime limited to T threads. pyg-lib: T DataLoader "
                "worker processes, one thread each (NeighborLoader's way of using cores).\n")
            add("| dataset | threads | ours (batches/s) | pyg-lib (batches/s) | ours / pyg-lib |")
            add("|---|---|---|---|---|")
            for g, table in cpu_rows:
                for t, row in table.items():
                    o, p = row.get("ours"), row.get("pyg")
                    add(f"| {g} | {t} | {fmt(o, 2)} | {fmt(p, 2)} | "
                        f"{fmt(o / p, 2, '×') if o and p else '–'} |")
            add("")

    if summary["e2e"]:
        add("## End-to-end training step (ogbn-products)\n")
        add("GraphSAGE 3×256, trimmed layers, 60 minibatches; mean sets epoch time.\n")
        add("| config | mean ms/step | median ms/step |\n|---|---|---|")
        pe = prev.get("e2e") or {}
        for cfg, r in sorted(summary["e2e"].items(), key=lambda kv: kv[1]["mean_ms"]):
            add(f"| {cfg} | {fmt(r['mean_ms'])}{delta(r['mean_ms'], pe.get(cfg, {}).get('mean_ms'))} "
                f"| {fmt(r['median_ms'])} |")
        ours = min((r["mean_ms"] for c, r in summary["e2e"].items() if c.startswith("ours")), default=None)
        pyg = min((r["mean_ms"] for c, r in summary["e2e"].items() if c.startswith("pyg")), default=None)
        if ours and pyg:
            add(f"\n**Ours vs PyG's best setting: {pyg / ours:.2f}× on mean step time.**")
        add("")

    if acc:
        add("## Accuracy parity (ogbn-products)\n")
        add("| sampler | seeds | val acc | test acc (50 K subset) | train s/epoch |\n|---|---|---|---|---|")
        for s in ("ours", "pyg"):
            if s in acc:
                a = acc[s]
                add(f"| {s} | {a['seeds']} | {a['val']:.4f} ± {a['val_sd']:.4f} | "
                    f"{a['test']:.4f} ± {a['test_sd']:.4f} | {a['epoch_s']:.1f} |")
        if "welch_p_test" in acc:
            add(f"\nWelch t-test: val p = {acc['welch_p_val']:.2f}, test p = {acc['welch_p_test']:.2f} "
                "(p well above 0.05 means no detectable difference).")
        add("")

    if summary["twitter"]:
        t = summary["twitter"]
        add("## Twitter-2010 (1.47 B edges)\n")
        o, p = t.get("ours", {}), t.get("pyg", {})
        add("| sampler | ms/batch | graph load (s) |\n|---|---|---|")
        if p:
            add(f"| pyg-lib (CPU, int32) | {fmt(p.get('median_ms'))} | {fmt(p.get('load_s'), 0)} |")
        if o:
            add(f"| ours v1 | {fmt(o.get('v1_median_ms'))} | {fmt(o.get('load_s'), 0)} |")
            add(f"| ours final | {fmt(o.get('final_median_ms'))} | {fmt(o.get('load_s'), 0)} |")
        if o and p:
            add(f"\nOurs vs pyg-lib: {p['median_ms'] / o['final_median_ms']:.2f}× (separate "
                "processes, same batches).")
        add("")

    add("## Notes\n")
    add("- Raw files for this iteration: `results/suite/iter_" + str(x) + "/`; logs: "
        f"`results/logs/suite{x}-*.log`.")
    add("- Percentages in brackets compare with the previous iteration. Absolute times from "
        "runs on different power states are not comparable; ratios within a run are.")
    out_md = SUITE / f"iteration_{x}.md"
    out_md.write_text("\n".join(L) + "\n")
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(out_md)


if __name__ == "__main__":
    main()
