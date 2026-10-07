#!/bin/bash
# Train the final AspGap-v3 model (configs/aspgap_dist.yaml) and evaluate it afterwards.
#   nohup scripts/run_aspgap_dist.sh aspgap_dist_r1 > logs/aspgap_dist_r1.log 2>&1 &
set -e
NAME=${1:-aspgap_dist_r1}
RUNS=${RUNS:-/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3/runs}
source /nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/bin/activate jepa311
cd "$(dirname "$0")/.."
python scripts/train.py --config configs/aspgap_dist.yaml --set run.name=$NAME run.out_dir=$RUNS
python scripts/evaluate.py --run $RUNS/$NAME
