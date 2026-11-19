#!/bin/sh
# Run a benchmark command inside a named gnnsample container with a hard timeout, logging
# everything to results/logs/<name>.log (tools/watch_runs.sh watches these logs).
# Usage: tools/run_logged.sh <name> <timeout-seconds> <command...>
set -u
# If the docker CLI is not on PATH, prepend its directory here, e.g.
# export PATH="/path/to/DockerDesktop/resources/bin:$PATH"
name=$1; limit=$2; shift 2
log=results/logs/$name.log
mkdir -p results/logs
echo "[run_logged] $(date -u +%FT%TZ) start: $* (timeout ${limit}s)" > "$log"
# Power state matters: on battery Windows throttles the CPU (pyg-lib ran 2.3x slower).
power=$(powershell.exe -NoProfile -Command '$b = Get-CimInstance Win32_Battery; if ($b) { if ($b.BatteryStatus -eq 2) { "AC" } else { "BATTERY" }; $b.EstimatedChargeRemaining }' 2>/dev/null | tr -d '\r' | paste -sd' ')
echo "[run_logged] power: ${power:-unknown} (absolute timings are comparable only on AC)" >> "$log"
MSYS_NO_PATHCONV=1 docker compose run --rm -T --name "gnnsample-$name" dev \
  timeout --signal=TERM --kill-after=30 "$limit" "$@" >> "$log" 2>&1
rc=$?
[ $rc -eq 124 ] && echo "[run_logged] TIMEOUT after ${limit}s" >> "$log"
echo "[run_logged] $(date -u +%FT%TZ) exit=$rc" >> "$log"
exit $rc
