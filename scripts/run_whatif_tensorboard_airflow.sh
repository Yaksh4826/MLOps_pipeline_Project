#!/usr/bin/env bash
# Lab2-style launch for COMP315 Group 8 (obesity) under Airflow dags path.
#
# Course lab: copy eval TFRecords under --logdir so WIT can open them; then TensorBoard.
# This model was trained on *transformed* examples — use Transform Split-eval, not CsvExampleGen.
#
# Usage (WSL):
#   conda activate tfx-env
#   Run from the git clone that *contains* scripts/ (e.g. ~/COMP315_Group8_project).
#   Point data at your Airflow copy:
#     export COMP315_PROJECT_ROOT=/root/airflow/dags/COMP315_Group8_project
#   chmod +x scripts/run_whatif_tensorboard_airflow.sh
#   ./scripts/run_whatif_tensorboard_airflow.sh
#
# TensorBoard in tfx-env expects *hyphenated* flags (--whatif-data-dir), not underscores.
#
# Override paths if needed:
#   export COMP315_PROJECT_ROOT=/root/airflow/dags/COMP315_Group8_project
#   export WIT_MODEL_DIR=/root/airflow/dags/.../Pusher/pushed_model/34
#   export WIT_EVAL_GLOB='.../Transform/transformed_examples/46/Split-eval/*.gz'

set -euo pipefail

PROJECT_ROOT="${COMP315_PROJECT_ROOT:-/root/airflow/dags/COMP315_Group8_project}"
OUT="${WIT_MODEL_DIR:-}"
if [[ -z "${OUT}" ]]; then
  OUT="$(ls -td "${PROJECT_ROOT}/tfx_airflow_runs/outputs/Pusher/pushed_model"/* 2>/dev/null | head -1 || true)"
fi
if [[ -z "${OUT}" || ! -d "${OUT}" ]]; then
  echo "ERROR: No Pusher model dir found under ${PROJECT_ROOT}/tfx_airflow_runs/outputs/Pusher/pushed_model/"
  echo "Set WIT_MODEL_DIR to the folder that contains saved_model.pb (e.g. .../pushed_model/34)."
  exit 1
fi

EVAL_SRC="${WIT_EVAL_GLOB:-}"
if [[ -z "${EVAL_SRC}" ]]; then
  # Newest transform eval shard (pick a run that matches your DAG run if inference looks wrong).
  EVAL_SRC="$(ls -t "${PROJECT_ROOT}"/tfx_airflow_runs/outputs/Transform/transformed_examples/*/Split-eval/*.gz 2>/dev/null | head -1 || true)"
fi
if [[ -z "${EVAL_SRC}" ]]; then
  echo "ERROR: No Transform Split-eval .gz found. Run the pipeline first or set WIT_EVAL_GLOB."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST="${OUT}/wit_eval.tfrecord.gz"
cp -f "${EVAL_SRC}" "${DEST}"

export WIT_SAVEDMODEL_PATH="${OUT}"

echo "Copied eval TFRecord:"
echo "  from: ${EVAL_SRC}"
echo "  to:   ${DEST}"
echo "WIT_SAVEDMODEL_PATH=${WIT_SAVEDMODEL_PATH}"
echo ""
echo "Starting TensorBoard (keep this terminal open). Open http://localhost:6006"
echo "Menu (top right) → What-If Tool → use custom prediction (see ReadMe / course adaption)."
echo ""

cd "${REPO_ROOT}"
# tfx-env: use the `tensorboard` entrypoint — some stacks break `python -m tensorboard`.
if command -v tensorboard >/dev/null 2>&1; then
  exec tensorboard --logdir="${OUT}" --bind_all \
    --whatif-data-dir="${OUT}" \
    --whatif-use-unsafe-custom-prediction="${REPO_ROOT}/scripts/wit_custom_predict.py"
else
  exec python -m tensorboard --logdir="${OUT}" --bind_all \
    --whatif-data-dir="${OUT}" \
    --whatif-use-unsafe-custom-prediction="${REPO_ROOT}/scripts/wit_custom_predict.py"
fi
