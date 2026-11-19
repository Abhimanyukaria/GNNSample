#!/bin/sh
# Watch gnnsample benchmark runs (logs in results/logs/, containers named gnnsample-<name>).
# Every INTERVAL seconds, one line per event:
#   STALL   log has not grown since the last check AND the container used < 2% CPU (the iGPU
#           can be busy with CPU at 0%, so CPU alone is not enough)
#   DONE    a log got its exit line
#   GONE    container stopped and still no exit line after a 30 s grace period
#   LOWMEM  Windows free memory < 1 GB: the running gnnsample containers are killed, since
#           on WSL2 GPU memory lives outside the VM and the VM gets paged out (D19)
# Exits by itself once no gnnsample container has run and no log was active for two checks,
# so a stopped Monitor does not leave this loop behind.
# Usage: tools/watch_runs.sh [interval_seconds]
interval=${1:-300}
# If the docker CLI is not on PATH, prepend its directory here, e.g.
# export PATH="/path/to/DockerDesktop/resources/bin:$PATH"
state=/tmp/gnnsample_watch.$$
mkdir -p "$state"
idle=0
while true; do
  free_mb=$(powershell.exe -NoProfile -Command '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)' 2>/dev/null | tr -d '\r')
  running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep '^gnnsample-' || true)
  if [ -n "$free_mb" ] && [ "$free_mb" -lt 1024 ] && [ -n "$running" ]; then
    echo "LOWMEM: Windows free memory ${free_mb} MB; killing $(echo $running | tr '\n' ' ')"
    echo "$running" | xargs -r docker kill >/dev/null 2>&1
  fi
  active=0
  for log in results/logs/*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log)
    size=$(wc -c < "$log")
    prev=$(cat "$state/$name.size" 2>/dev/null || echo -1)
    echo "$size" > "$state/$name.size"
    if grep -q '^\[run_logged\].*exit=' "$log"; then
      if [ "$prev" != "-1" ] && [ ! -f "$state/$name.done" ]; then
        echo "DONE $name: $(grep '^\[run_logged\].*exit=' "$log" | tail -1 | sed 's/.*exit=/exit=/') | last: $(grep -v '^\[run_logged\]' "$log" | tail -1 | cut -c1-120)"
      fi
      touch "$state/$name.done"
      continue
    fi
    active=1
    cpu=$(docker stats --no-stream --format '{{.CPUPerc}}' "gnnsample-$name" 2>/dev/null | tr -d '%')
    if [ -z "$cpu" ]; then
      sleep 30
      if ! grep -q '^\[run_logged\].*exit=' "$log" && [ ! -f "$state/$name.gone" ]; then
        touch "$state/$name.gone"
        echo "GONE $name: container stopped and no exit line after 30 s (killed?)"
      fi
      continue
    fi
    if [ "$size" = "$prev" ] && awk -v c="$cpu" 'BEGIN { exit !(c < 2) }'; then
      echo "STALL $name: no log output for ${interval}s and CPU ${cpu}% | last: $(tail -1 "$log" | cut -c1-120)"
    fi
  done
  if [ "$active" = 0 ] && [ -z "$running" ]; then idle=$((idle + 1)); else idle=0; fi
  [ "$idle" -ge 2 ] && { echo "watcher: nothing running, exiting"; exit 0; }
  sleep "$interval"
done
