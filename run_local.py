#!/usr/bin/env python3
"""Run the TFX pipeline locally (no Airflow) — same create_pipeline as the DAG."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tfx.orchestration.local.local_dag_runner import LocalDagRunner

from obesity_tfx import create_pipeline


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    pipeline_root = os.path.join(root, "tfx_pipeline_runs")
    data_root = os.path.join(root, "data", "obesity")
    metadata_path = os.path.join(pipeline_root, "metadata", "metadata.db")
    serving_model_dir = os.path.join(root, "serving_model")

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    os.makedirs(serving_model_dir, exist_ok=True)

    pipeline = create_pipeline(
        pipeline_name="obesity_ml_pipeline",
        pipeline_root=pipeline_root,
        data_root=data_root,
        metadata_path=metadata_path,
        serving_model_dir=serving_model_dir,
    )
    LocalDagRunner().run(pipeline)


if __name__ == "__main__":
    main()
