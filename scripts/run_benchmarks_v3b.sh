#!/usr/bin/env bash
# Extra ablations queued AFTER run_benchmarks_v3.sh (PID below) has exited:
#   aspgap110_aug_r2         aspgap110 + noise-matched augmentation (p=0.5)
#   pixel_fusion_snronly_r2  pixel_fusion with side scalars [log_snr550] only
# Code runs from the p1 WORKTREE (so the main tree stays untouched while the
# first queue is alive); data and run outputs live in the MAIN tree.
set -u
WAIT_PID=${WAIT_PID:-1634664}
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/envs/jepa311/bin/python
MAIN=/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3
WT=/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3_p1
ROOT=$MAIN/data/processed/aspcap_v0.6.0
OUT=$MAIN/runs

while kill -0 "$WAIT_PID" 2>/dev/null; do
  echo "$(date '+%F %T') waiting for PID $WAIT_PID"
  sleep 300
done
echo "=== $(date '+%F %T') PID $WAIT_PID gone; starting v3b queue ==="

cd "$WT"   # relative ../aspgap/... paths (scalers, specclip ckpt) resolve as in the main tree

run() {
  name=$1; cfg=$2
  echo "=== $(date '+%F %T') starting $name ==="
  $PY scripts/train.py --config configs/$cfg.yaml --set "run.name=$name" \
      "run.out_dir=$OUT" "data.root=$ROOT" 2>&1
  echo "=== $(date '+%F %T') evaluating $name (incl. degraded G' 15/16/17/17.5) ==="
  $PY scripts/evaluate.py --run $OUT/$name --splits iid_test sky_test \
      --degrade-g 15 16 17 17.5 2>&1
  echo "=== $(date '+%F %T') finished $name ==="
}

run aspgap110_aug_r2        aspgap110_aug
run pixel_fusion_snronly_r2 pixel_fusion_snronly
echo "=== $(date '+%F %T') ALL DONE (v3b) ==="
