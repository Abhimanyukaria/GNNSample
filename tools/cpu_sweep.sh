#!/bin/sh
# CPU core-scaling sweep (runs inside the dev container): our CPU sampler vs pyg-lib workers
# at equal thread counts, alternating, R rounds. Usage: tools/cpu_sweep.sh OUT.jsonl "graphs" "threads" R
out=$1; graphs=${2:-"products orkut"}; threads=${3:-"1 2 4 6 8 10 12 14"}; rounds=${4:-2}
for r in $(seq 1 "$rounds"); do
  for g in $graphs; do
    for t in $threads; do
      python bench/cpu_scaling.py --graph "$g" --sampler ours --threads "$t" --json-out "$out"
      python bench/cpu_scaling.py --graph "$g" --sampler pyg --threads "$t" --json-out "$out"
    done
  done
done
