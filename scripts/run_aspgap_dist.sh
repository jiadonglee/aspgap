#!/bin/bash
# Train a config and evaluate it afterwards (generic: pick the config with CONFIG=).
#   nohup scripts/run_aspgap_dist.sh aspgap_dist_r3 > logs/aspgap_dist_r3.log 2>&1 &                       # APOGEE only
#   CONFIG=configs/aspgap_dist_ext2_cn.yaml nohup scripts/run_aspgap_dist.sh aspgap_dist_ext2_cn > logs/ext2_cn.log 2>&1 &   # final model
#   (needs data/processed/aspcap_v0.6.0_ext2_cn: python scripts/build_cn_labels.py)
# Shared nodes: use at most half of nproc (dataloader workers included) and set OMP_NUM_THREADS etc. before calling.
set -e
NAME=${1:-aspgap_dist_r1}
CONFIG=${CONFIG:-configs/aspgap_dist.yaml}
RUNS=${RUNS:-/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3/runs}
source /nexus/posix0/MIA-astro-env/hxr/jdli/miniforge3/bin/activate jepa311
cd "$(dirname "$0")/.."
python scripts/train.py --config $CONFIG --set run.name=$NAME run.out_dir=$RUNS
python scripts/evaluate.py --run $RUNS/$NAME
