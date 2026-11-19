#!/bin/sh
# Decode the LAW Twitter-2010 transposed BVGraph into data/csc/twitter (in-neighbour CSC),
# in the converter image, with a hard timeout and a log that tools/watch_runs.sh follows.
set -u
# If the docker CLI is not on PATH, prepend its directory here, e.g.
# export PATH="/path/to/DockerDesktop/resources/bin:$PATH"
log=results/logs/convert-twitter.log
echo "[run_logged] $(date -u +%FT%TZ) start: ToCsc twitter-2010-t (timeout 5400s)" > "$log"
MSYS_NO_PATHCONV=1 docker run --rm --name gnnsample-convert-twitter \
  -v "$(pwd)/data/raw/twitter:/in:ro" -v "$(pwd)/data/csc/twitter:/out" \
  --entrypoint timeout gnnsample-webgraph 5400 \
  java -Xmx2g -cp "/opt/webgraph:/opt/webgraph/lib/*" ToCsc /in/twitter-2010-t /out >> "$log" 2>&1
rc=$?
echo "[run_logged] $(date -u +%FT%TZ) exit=$rc" >> "$log"
exit $rc
