#!/usr/bin/env bash
# Queue v3e (replaces the v3c / v3d waiters, which were killed).
# Waits until the v3b waiter (PID 1644988) AND the orphaned pixel_fusion_p1_r2 training that v3c
# had already started (PID 1835292; its waiter was killed, so nobody evaluates it) have exited,
# then runs, from the p1 WORKTREE into the main runs/ (names *_r2), each followed by
# evaluate.py --splits iid_test sky_test --degrade-g 15 16 17 17.5:
#   mlp343_aug, aspgap110_aug_p1, mlp343_aug_p1, pixel_fusion_noaug,
#   pixel_fusion_p1 (evaluate-only if the orphan finished training, else full run),
#   pixel_fusion_p1_auxonly, pixel_fusion_p1_condonly
# A failed run never stops the queue.
set -u
WAIT_PIDS=${WAIT_PIDS:-"1644988 1835292"}
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/envs/jepa311/bin/python
MAIN=/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3
WT=/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3_p1
ROOT=$MAIN/data/processed/aspcap_v0.6.0
OUT=$MAIN/runs

for pid in $WAIT_PIDS; do
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date '+%F %T') waiting for PID $pid"
    sleep 300
  done
done
echo "=== $(date '+%F %T') wait PIDs ($WAIT_PIDS) gone; starting v3e queue ==="

cd "$WT"   # relative ../aspgap/... paths (scalers, specclip ckpt) resolve as in the main tree

train() {
  name=$1; cfg=$2
  echo "=== $(date '+%F %T') starting $name ==="
  $PY scripts/train.py --config configs/$cfg.yaml --set "run.name=$name" \
      "run.out_dir=$OUT" "data.root=$ROOT" 2>&1 || echo "!!! train.py FAILED for $name (exit $?)"
}
evalrun() {
  name=$1
  if [ ! -f "$OUT/$name/ckpt_best.pt" ]; then
    echo "!!! no checkpoint for $name, skipping evaluation"; return
  fi
  echo "=== $(date '+%F %T') evaluating $name (incl. degraded G' 15/16/17/17.5) ==="
  $PY scripts/evaluate.py --run $OUT/$name --splits iid_test sky_test \
      --degrade-g 15 16 17 17.5 2>&1 || echo "!!! evaluate.py FAILED for $name (exit $?)"
  echo "=== $(date '+%F %T') finished $name ==="
}
run() { train "$1" "$2"; evalrun "$1"; }

run mlp343_aug_r2         mlp343_aug
run aspgap110_aug_p1_r2   aspgap110_aug_p1
run mlp343_aug_p1_r2      mlp343_aug_p1
run pixel_fusion_noaug_r2 pixel_fusion_noaug
if [ -f "$OUT/pixel_fusion_p1_r2/ckpt_last.pt" ]; then
  echo "=== $(date '+%F %T') pixel_fusion_p1_r2 already trained (v3c orphan); evaluating only ==="
  evalrun pixel_fusion_p1_r2
else
  run pixel_fusion_p1_r2 pixel_fusion_p1
fi
run pixel_fusion_p1_auxonly_r2  pixel_fusion_p1_auxonly
run pixel_fusion_p1_condonly_r2 pixel_fusion_p1_condonly
echo "=== $(date '+%F %T') ALL DONE ==="
