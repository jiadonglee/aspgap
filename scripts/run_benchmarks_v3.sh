#!/usr/bin/env bash
# Sequential benchmark queue, round 2 (_r2): XP-valid star set, bf16 training,
# fp32 eval, degraded-G' evaluation after every run. One GPU, shared.
set -u
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/envs/jepa311/bin/python
cd /nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3
ROOT=data/processed/aspcap_v0.6.0

run() {
  name=$1; cfg=$2
  echo "=== $(date '+%F %T') starting $name ==="
  $PY scripts/train.py --config configs/$cfg.yaml --set "run.name=$name" "data.root=$ROOT" 2>&1
  echo "=== $(date '+%F %T') evaluating $name (incl. degraded G' 15/16/17/17.5) ==="
  $PY scripts/evaluate.py --run runs/$name --splits iid_test sky_test \
      --degrade-g 15 16 17 17.5 2>&1
  echo "=== $(date '+%F %T') finished $name ==="
}

run pixel_fusion_r2       pixel_fusion
run aspgap110_r2          aspgap110
run specclip_ft_r2        specclip_ft
run pixel_fusion_noaug_r2 pixel_fusion_noaug
run specclip_frozen_r2    specclip_frozen
run pixel_fusion_phot_r2  pixel_fusion_phot
run scratch_r2            scratch
echo "=== $(date '+%F %T') ALL DONE ==="
