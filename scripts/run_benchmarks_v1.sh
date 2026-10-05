#!/usr/bin/env bash
# Sequential benchmark queue: one GPU, four configs, v1 names.
set -u
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/envs/jepa311/bin/python
cd /nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3

run() {
  name=$1; shift
  echo "=== $(date '+%F %T') starting $name ==="
  $PY scripts/train.py "$@" --set "run.name=$name" 2>&1
  echo "=== $(date '+%F %T') finished $name ==="
}

run specclip_frozen_v1 --config configs/specclip_frozen.yaml
run specclip_ft_v1     --config configs/specclip_ft.yaml
run scratch_v1         --config configs/scratch.yaml
run aspgap110_v1       --config configs/aspgap110.yaml
echo "=== $(date '+%F %T') ALL DONE ==="
