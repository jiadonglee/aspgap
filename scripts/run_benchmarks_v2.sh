#!/usr/bin/env bash
# v2 queue: stats-token pooling (continuum info) + Stage C full FT.
set -u
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/envs/jepa311/bin/python
cd /nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3

run() {
  name=$1; shift
  echo "=== $(date '+%F %T') starting $name ==="
  $PY scripts/train.py "$@" --set "run.name=$name" 2>&1
  echo "=== $(date '+%F %T') finished $name ==="
}

run specclip_frozen_v2 --config configs/specclip_frozen_v2.yaml
run specclip_ftfull_v2 --config configs/specclip_ftfull_v2.yaml
echo "=== $(date '+%F %T') V2 ALL DONE ==="
