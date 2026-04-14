#!/usr/bin/env bash
# One-shot: sync scripts → Airflow dag copy, copy eval TFRecords next to SavedModel, open TensorBoard + What-If.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIRFLOW="${COMP315_PROJECT_ROOT:-/root/airflow/dags/COMP315_Group8_project}"

# Use tfx-env if TF not on PATH (matches your machine)
if ! python -c "import tensorboard" 2>/dev/null; then
  for CONDA_SH in /root/anaconda3/etc/profile.d/conda.sh "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "$CONDA_SH" ]]; then
      # shellcheck source=/dev/null
      source "$CONDA_SH"
      conda activate tfx-env
      break
    fi
  done
fi

mkdir -p "$AIRFLOW/scripts"
rsync -a "$HERE/scripts/" "$AIRFLOW/scripts/"

export COMP315_PROJECT_ROOT="$AIRFLOW"
exec bash "$HERE/scripts/run_whatif_tensorboard_airflow.sh"
