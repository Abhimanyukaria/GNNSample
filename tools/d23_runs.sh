#!/bin/sh
# D23 run list: graph reordering, hub cache hints, cross-minibatch schedule.
# Runs one step at a time (the GPU is shared; never two benchmarks at once), each through
# run_logged.sh with a hard timeout and a log in results/logs/d23-*.log. Start a watcher
# alongside:  sh tools/watch_runs.sh 300
# Usage: sh tools/d23_runs.sh [step...]   steps: test reorder locality ab multi (default: all)
# Disk: the reorderings add ~7 GB under data/csc/ (4 numberings x products, LiveJournal, Orkut).
set -u
cd "$(dirname "$0")/.."
steps=${*:-test reorder locality ab multi}
GRAPHS="products livejournal orkut"
METHODS="degree hubsort rcm random"
run() { sh tools/run_logged.sh "$@" || echo "[d23] step $1 exited $?"; }
has() { case " $steps " in *" $1 "*) return 0 ;; esac; return 1; }

# 1. Bit-identity: every new path must match the reference (tests/test_reorder.py + option masks).
has test && run d23-test 2400 python -m pytest -q tests/

# 2. Renumbered copies of each graph (CPU only; skipped when already written).
if has reorder; then
  for g in $GRAPHS; do for m in $METHODS; do
    [ -f "data/csc/$g@$m/meta.json" ] || run "d23-reorder-$g-$m" 3600 python -m gnnsample.reorder --graph "$g" --method "$m"
  done; done
fi

# 3. Hardware-independent counters: lines touched, hub share, cross-K reuse.
if has locality; then
  for g in $GRAPHS; do
    run "d23-locality-$g" 2400 python bench/locality_stats.py --graph "$g" \
      --variants "$g@degree,$g@hubsort,$g@rcm,$g@random"
  done
fi

# 4. Timing, today's sampler: original vs each numbering, +/- hub hints (paired, one process).
#    Orkut holds one 0.94 GB graph copy per variant on the device, so fewer variants there.
if has ab; then
  run d23-ab-products 3600 python bench/reorder_ab.py --graph products --methods degree,hubsort,rcm,random --hub-mb 1,4,16
  run d23-ab-livejournal 3600 python bench/reorder_ab.py --graph livejournal --methods degree,hubsort,rcm,random --hub-mb 1,4,16
  run d23-ab-orkut 3600 python bench/reorder_ab.py --graph orkut --methods degree,rcm --hub-mb 4
fi

# 5. All three designs kept side by side (today / batched / megakernel), now with the
#    locality schedule, on the original and the degree numbering (+ hub hints). One mode
#    group per process, as in D22 (Orkut exhausted Windows memory with everything at once).
if has multi; then
  for g in $GRAPHS; do
    for v in "$g" "$g@degree"; do
      hub=""; [ "$v" != "$g" ] && hub="--hub-mb 4"
      tag=$(echo "$v" | tr '@' '-')
      run "d23-multi-batched-$tag" 3600 python bench/multi_ab.py --graph "$v" --batched 8 --sched 4,8,16 --mega "" $hub
      run "d23-multi-mega-$tag" 3600 python bench/multi_ab.py --graph "$v" --batched "" --mega 16 $hub
    done
  done
fi
echo "[d23] done: results/locality_stats.jsonl, results/reorder_ab.jsonl, results/multi_ab.jsonl"
