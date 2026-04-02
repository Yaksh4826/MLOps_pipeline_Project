from datetime import datetime
from airflow import DAG
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from pipeline.pipeline import create_pipeline

PIPELINE_DAG_NAME = "obesity_data_validation_pipeline"

default_args = {
    "start_date": datetime(2026, 1, 1),
}

with DAG(
    dag_id=PIPELINE_DAG_NAME,
    default_args=default_args,
    schedule_interval=None
) as dag:

    AirflowDagRunner().run(
        create_pipeline()
    )