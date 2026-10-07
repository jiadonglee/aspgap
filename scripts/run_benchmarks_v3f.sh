#!/usr/bin/env bash
# Queue v3f: takes over from v3e (whose bash shell is killed AFTER this waiter is up; the
# running aspgap110_aug_p1_r2 training child is left alone and merely waited for).
# Waits for WAIT_PIDS (the running aspgap110_aug_p1_r2 train.py), then, from the p1 WORKTREE
# into the main runs/ (names *_r2), runs in this order (each train followed by
# evaluate.py --splits iid_test sky_test --degrade-g 15 16 17 17.5):
#   aspgap110_aug_p1_r2 (evaluate only: v3e is gone so nobody else evaluates it)
#   mlp343_aug_p1_r2
#   aspgap110_aug_rand_r2   (A: random sampler)
#   aspgap110_aug_v1lat_r2  (B: v1 latent distillation)
#   scripts/compare_subsets.py (v1 / aug_r2 / A / B on bright giants & never-in-DR17)
#   pixel_fusion_noaug_r2, pixel_fusion_p1_r2 (evaluate-only if trained), pixel_fusion_p1_auxonly_r2,
#   pixel_fusion_p1_condonly_r2
# Idempotent: a run whose log.jsonl has the train.py "done" event is not re-trained; a run with
# summary.md (evaluate.py writes it last) is not re-evaluated. A failed run never stops the queue.
set -u
set -o pipefail
WAIT_PIDS=${WAIT_PIDS:-"1885320"}
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/envs/jepa311/bin/python
MAIN=/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3
WT=/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3_p1
ROOT=$MAIN/data/processed/aspcap_v0.6.0
OUT=$MAIN/runs

for pid in $WAIT_PIDS; do
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date '+%F %T') waiting for PID $pid"
    sleep 60
  done
done
echo "=== $(date '+%F %T') wait PIDs ($WAIT_PIDS) gone; starting v3f queue ==="

cd "$WT"   # relative ../aspgap/... paths (scalers, specclip ckpt) resolve as in the main tree

trained()   { grep -q '"event": "done"' "$OUT/$1/log.jsonl" 2>/dev/null; }
evaluated() { [ -f "$OUT/$1/summary.md" ]; }

train() {
  name=$1; cfg=$2
  if trained "$name"; then
    echo "=== $(date '+%F %T') $name already trained (done event in log.jsonl); skipping training ==="; return
  fi
  echo "=== $(date '+%F %T') starting $name ==="
  $PY scripts/train.py --config configs/$cfg.yaml --set "run.name=$name" \
      "run.out_dir=$OUT" "data.root=$ROOT" 2>&1 || echo "!!! train.py FAILED for $name (exit $?)"
}
evalrun() {
  name=$1
  if evaluated "$name"; then
    echo "=== $(date '+%F %T') $name already evaluated (summary.md exists); skipping ==="; return
  fi
  if [ ! -f "$OUT/$name/ckpt_best.pt" ]; then
    echo "!!! no checkpoint for $name, skipping evaluation"; return
  fi
  echo "=== $(date '+%F %T') evaluating $name (incl. degraded G' 15/16/17/17.5) ==="
  $PY scripts/evaluate.py --run $OUT/$name --splits iid_test sky_test \
      --degrade-g 15 16 17 17.5 2>&1 || echo "!!! evaluate.py FAILED for $name (exit $?)"
  echo "=== $(date '+%F %T') finished $name ==="
}
run() { train "$1" "$2"; evalrun "$1"; }

run aspgap110_aug_p1_r2      aspgap110_aug_p1        # already trained by v3e's child -> evaluate only
run mlp343_aug_p1_r2         mlp343_aug_p1
run aspgap110_aug_rand_r2    aspgap110_aug_rand
run aspgap110_aug_v1lat_r2   aspgap110_aug_v1lat
echo "=== $(date '+%F %T') compare_subsets (v1 / aug_r2 / A / B) ==="
$PY scripts/compare_subsets.py --out $OUT/compare_subsets_rand_v1lat 2>&1 \
  || echo "!!! compare_subsets.py FAILED (exit $?)"
echo "=== $(date '+%F %T') compare_subsets done ==="
run pixel_fusion_noaug_r2    pixel_fusion_noaug
run pixel_fusion_p1_r2       pixel_fusion_p1         # trained by the v3c orphan -> evaluate only
run pixel_fusion_p1_auxonly_r2  pixel_fusion_p1_auxonly
run pixel_fusion_p1_condonly_r2 pixel_fusion_p1_condonly
echo "=== $(date '+%F %T') ALL DONE ==="
