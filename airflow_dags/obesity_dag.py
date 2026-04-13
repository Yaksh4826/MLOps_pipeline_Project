import os
import sys

# Before TFX/TensorFlow: limit BLAS/TF threads so training/inference do not oversubscribe RAM
# (common cause of OOM → SIGKILL on small hosts / WSL).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from datetime import datetime

from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from obesity_tfx import create_pipeline

_PROJECT_ROOT = parent_dir
_PIPELINE_ARTIFACT_ROOT = os.path.join(_PROJECT_ROOT, "tfx_airflow_runs")

AIRFLOW_CONFIG = {
    "schedule_interval": None,
    "start_date": datetime(2024, 1, 1),
    "catchup": False,
}

DAG = AirflowDagRunner(config=AIRFLOW_CONFIG).run(
    create_pipeline(
        pipeline_name="obesity_ml_pipeline",
        pipeline_root=os.path.join(_PIPELINE_ARTIFACT_ROOT, "outputs"),
        data_root=os.path.join(_PROJECT_ROOT, "data", "obesity"),
        metadata_path=os.path.join(_PIPELINE_ARTIFACT_ROOT, "metadata", "metadata.db"),
        # Blessed SavedModel copy for TensorFlow Serving / What-If (project root, not under pipeline artifacts).
        serving_model_dir=os.path.join(_PROJECT_ROOT, "serving_model"),
    )
)
