#!/usr/bin/env bash
# Run ONE training experiment with the paper's settings (default: RATS, Sampling-Only).
#
# Usage (from anywhere, inside the activated virtual environment):
#   bash scripts/run_train.sh <dataset> [seed] [extra "--set KEY=VALUE" overrides ...]
#
#   <dataset>  cub200 | cars196 | sop | veriwild   (uses configs/<dataset>.yaml)
#   [seed]     random seed, default 1
#
# Examples: 
#   bash scripts/run_train.sh cub200
#   bash scripts/run_train.sh sop 2
#
#   # change a config value for this run (nested keys use a dot):
#   bash scripts/run_train.sh cars196 1 --set rl.method=A2C
#   bash scripts/run_train.sh cub200 1 --set rl.region_step=0.15 --set epochs=60
#
#   # Joint variant (adaptive margin):
#   EXP_NAME=rats_joint bash scripts/run_train.sh sop 1 --set rl.adaptive_margin=True
#
#   # static baseline instead of RATS (batch_hard | semi_hard | distance_weighted):
#   EXP_NAME=batch_hard bash scripts/run_train.sh sop 1 \
#       --set rl.enabled=False --set sampling_strategy=batch_hard
#
# The script just calls train.py. The equivalent direct command is:
#   python train.py --config configs/cub200.yaml --set seed=1
# Values can also be edited permanently in configs/<dataset>.yaml.

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:?Usage: bash scripts/run_train.sh <cub200|cars196|sop|veriwild> [seed] [--set KEY=VALUE ...]}"
SEED="${2:-1}"
shift $(( $# < 2 ? $# : 2 ))

EXP_NAME="${EXP_NAME:-rats}"
TAG="${DATASET}/${EXP_NAME}/seed${SEED}"

python train.py \
    --config "configs/${DATASET}.yaml" \
    --set "seed=${SEED}" \
    --set "experiment_name=${EXP_NAME}" \
    --set "output_dir=runs/${TAG}" \
    --set "checkpoint_dir=checkpoints/${TAG}" \
    "$@"