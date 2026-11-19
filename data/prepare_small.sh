#!/bin/sh
# Downloads and converts the laptop-scale datasets (approved 2026-10-24).
set -e
for d in products livejournal orkut; do python data/prepare.py "$d"; done
