#!/bin/sh
# One-command benchmark suite. Each run is a new iteration: results/suite/iteration_<x>.md
# (x = previous + 1), with raw files in results/suite/iter_<x>/ and logs in
# results/logs/suite<x>-*.log. Run from the project root in Git Bash.
#
#   sh tools/bench_suite.sh                              # products, livejournal, orkut
#   sh tools/bench_suite.sh -d "products orkut"          # choose datasets
#   sh tools/bench_suite.sh --accuracy                   # + accuracy parity (~55 min)
#   sh tools/bench_suite.sh --twitter                    # + Twitter-2010 scale run (~10 min)
#   sh tools/bench_suite.sh --cpu-threads "1 2 4 8 14"   # CPU thread counts (default 1 4 14)
#   sh tools/bench_suite.sh --quick                      # fewer batches, for a smoke test
#
# Steps (each in its own container with a hard timeout; a failed step is recorded and the
# suite continues): build, tests, then per dataset the GPU sampler vs v1 and pyg-lib, GPU
# stage times and the CPU sampler vs pyg-lib at equal thread counts; then the end-to-end
# training step on products (ours vs PyG with 0 and 2 workers); optional accuracy and Twitter.
# A watcher runs alongside and kills the containers if Windows runs low on memory.
set -u
# If the docker CLI is not on PATH, prepend its directory here, e.g.
# export PATH="/path/to/DockerDesktop/resources/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1

datasets="products livejournal orkut"; accuracy=0; twitter=0; cpu_threads="1 4 14"; quick=0
while [ $# -gt 0 ]; do
  case "$1" in
    -d|--datasets) datasets=$2; shift ;;
    --accuracy) accuracy=1 ;;
    --twitter) twitter=1 ;;
    --cpu-threads) cpu_threads=$2; shift ;;
    --quick) quick=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
batches=40; e2e_batches=60; [ "$quick" = 1 ] && { batches=10; e2e_batches=20; }

mkdir -p results/suite
last=$(ls results/suite 2>/dev/null | sed -n 's/^iteration_\([0-9]*\)\.md$/\1/p' | sort -n | tail -1)
x=$(( ${last:-0} + 1 ))
dir=results/suite/iter_$x
mkdir -p "$dir"
power() {
  powershell.exe -NoProfile -Command '$b = Get-CimInstance Win32_Battery; if ($b) { if ($b.BatteryStatus -eq 2) { "AC" } else { "BATTERY" }; $b.EstimatedChargeRemaining }' 2>/dev/null | tr -d '\r' | paste -sd' '
}
start_s=$(date +%s); start=$(date -u +%FT%TZ)
commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)
dirty=false; [ -n "$(git status --porcelain 2>/dev/null)" ] && dirty=true
power_start=$(power)
echo "== gnnsample suite iteration $x: datasets [$datasets], accuracy=$accuracy, twitter=$twitter, power $power_start"

sh tools/watch_runs.sh 60 > "$dir/watch.log" 2>&1 &
watcher=$!

: > "$dir/steps.tsv"
step() {  # step NAME TIMEOUT_S CMD...
  name=$1; limit=$2; shift 2
  t0=$(date +%s)
  sh tools/run_logged.sh "suite$x-$name" "$limit" "$@"
  rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" >> "$dir/steps.tsv"
  echo "   $name: exit $rc ($(( $(date +%s) - t0 )) s)"
}

step build 1200 sh -c "cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx >/dev/null && cmake --build build"
step tests 1800 sh -c "python -m pytest -q tests/ 2>&1 | tail -1 | tee $dir/tests.txt"

for g in $datasets; do
  step "gpu-$g" 1800 python bench/ablate.py --graph "$g" --variants 8169,0 --batches "$batches" --json-out "$dir/ablate_$g.jsonl"
  step "stages-$g" 1200 python bench/sampler_stages.py --tag "suite$x" --graphs "$g" --options 8169 --batches "$batches" --json-out "$dir/stages_$g.jsonl"
  step "cpu-$g" 2400 sh tools/cpu_sweep.sh "$dir/cpu_$g.jsonl" "$g" "$cpu_threads" 1
done

case " $datasets " in *" products "*)
  for c in ours-batched ours-overlap pyg-w0 pyg-w2; do
    step "e2e-$c" 1500 python bench/train_e2e.py --config "$c" --batches "$e2e_batches" --json-out "$dir/e2e.jsonl"
  done
  if [ "$accuracy" = 1 ]; then
    for seed in 0 1 2; do for s in ours pyg; do
      step "acc-$s-$seed" 2400 python bench/accuracy.py --sampler "$s" --seed "$seed" --out-dir "$dir/accuracy"
    done; done
  fi ;;
esac

if [ "$twitter" = 1 ]; then
  if [ -f data/csc/twitter/meta.json ]; then
    step twitter-pyg 2400 python bench/scale.py --graph twitter --mode pyg --batches "$batches" --json-out "$dir/scale_twitter.jsonl"
    step twitter-ours 2400 python bench/scale.py --graph twitter --mode ours --batches "$batches" --json-out "$dir/scale_twitter.jsonl"
  else
    echo "   twitter: data/csc/twitter missing (see tools/convert_twitter.sh); skipped"
  fi
fi

kill "$watcher" 2>/dev/null
end_s=$(date +%s)
cat > "$dir/meta.json" <<EOF
{"iteration": $x, "start": "$start", "end": "$(date -u +%FT%TZ)", "minutes": $(( (end_s - start_s) / 60 )),
 "commit": "$commit", "dirty": $dirty, "power_start": "$power_start", "power_end": "$(power)",
 "datasets": [$(printf '"%s",' $datasets | sed 's/,$//')], "accuracy": $accuracy, "twitter": $twitter,
 "cpu_threads": "$cpu_threads", "quick": $quick}
EOF
sh tools/run_logged.sh "suite$x-report" 600 python bench/suite_report.py --iter "$x"
echo "== report: results/suite/iteration_$x.md"
