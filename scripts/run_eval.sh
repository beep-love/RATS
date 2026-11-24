#!/usr/bin/env bash
# Evaluate ONE trained model (best checkpoint of a run) on a benchmark.
#
# Usage (from anywhere, inside the activated virtual environment):
#   bash scripts/run_eval.sh <dataset> [seed]
#
#   <dataset>  cub200 | cars196 | sop | veriwild
#   [seed]     seed of the training run, default 1
#
# Examples:
#   bash scripts/run_eval.sh cub200
#   bash scripts/run_eval.sh sop 2
#   bash scripts/run_eval.sh veriwild 1          # small / medium / large test splits
#
#   # a run trained with another name (see EXP_NAME in run_train.sh):
#   EXP_NAME=batch_hard bash scripts/run_eval.sh sop 1
#
#   # an explicit checkpoint, e.g. a downloaded one:
#   CKPT=path/to/best_model.pth bash scripts/run_eval.sh cars196
#
# By default the script picks the newest best_model_*.pth in
#   checkpoints/<dataset>/<EXP_NAME>/seed<seed>/        (EXP_NAME defaults to "rats")
# The equivalent direct commands are:
#   python evaluate.py --dataset CUB200 --checkpoint <path>          # CUB200 | CARS196 | SOP
#   python evaluate_veriwild.py --data_root data/VeRi-Wild --checkpoint <path>

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:?Usage: bash scripts/run_eval.sh <cub200|cars196|sop|veriwild> [seed]}"
SEED="${2:-1}"
EXP_NAME="${EXP_NAME:-rats}"

if [[ -z "${CKPT:-}" ]]; then
    CKPT_DIR="checkpoints/${DATASET}/${EXP_NAME}/seed${SEED}"
    CKPT="$(ls -t "${CKPT_DIR}"/best_model_*.pth 2>/dev/null | head -n 1 || true)"
    [[ -n "${CKPT}" ]] || { echo "No best_model_*.pth found in ${CKPT_DIR} (set CKPT=<path> to use another file)" >&2; exit 1; }
fi
echo "Checkpoint: ${CKPT}"

case "${DATASET}" in
    cub200)   python evaluate.py --dataset CUB200  --checkpoint "${CKPT}" ;;
    cars196)  python evaluate.py --dataset CARS196 --checkpoint "${CKPT}" ;;
    sop)      python evaluate.py --dataset SOP     --checkpoint "${CKPT}" ;;
    veriwild) python evaluate_veriwild.py --data_root data/VeRi-Wild --checkpoint "${CKPT}" ;;
    *) echo "Unknown dataset '${DATASET}'. Use cub200, cars196, sop or veriwild." >&2; exit 1 ;;
esac