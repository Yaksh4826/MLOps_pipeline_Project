import os, sys
from datetime import datetime
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner

# Add project root to path so pipeline.py can be imported
sys.path.insert(0, "/root/airflow/dags/COMP315_Group8_project")
from pipeline.pipeline import create_pipeline

BASE = "/root/airflow/dags/COMP315_Group8_project"

AIRFLOW_CONFIG = {
    "schedule_interval": None,
    "start_date": datetime(2024, 1, 1),
    "catchup": False,
}

DAG = AirflowDagRunner(config=AIRFLOW_CONFIG, enable_cache=False).run(
    create_pipeline(
        pipeline_name="obesity_data_validation_pipeline",
        pipeline_root=os.path.join(BASE, "outputs"),
        data_root=os.path.join(BASE, "data", "obesity"),
        metadata_path=os.path.join(BASE, "metadata", "metadata.db"),
    )
)