"""Generate Visualization.ipynb (TFDV + TFMA) for COMP315 obesity pipeline."""
import json
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Visualization.ipynb",
)

cells = []

cells.append(
    new_markdown_cell(
        """# Obesity ML pipeline — visualization (COMP315)

End-to-end **MLOps** artifacts from **TFX** (`obesity_tfx/tfx_pipeline.py`), orchestrated by **Airflow** (`airflow_dags/obesity_dag.py`). This notebook **inspects** each stage’s outputs.

| Step | Component | In this notebook |
|---|---|---|
| 1 | `CsvExampleGen` | Step 1 |
| 2 | `StatisticsGen` | Step 2 |
| 3 | `SchemaGen` | Step 3 |
| 4 | `ExampleValidator` | Step 4 |
| 5 | `Transform` | Step 5 |
| 6 | `Trainer` | Step 6 |
| 8 | `Evaluator` | Step 8 |
| 9 | `Pusher` | Step 9 |

**Prerequisite:** run **`obesity_ml_pipeline`** so `tfx_airflow_runs/outputs/` exists.

**Environment:** **`tfx-env`** (`tensorflow-model-analysis`, `tensorflow-data-validation`, `tfx`)."""
    )
)

cells.append(
    new_code_cell(
        """import pathlib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import cast

import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
from google.protobuf.json_format import ParseDict
from google.protobuf.message import Message
from tensorflow_model_analysis.proto import config_pb2

# --- locate pipeline outputs (Airflow deployment or local repo clone) ---
def find_outputs_dir():
    roots = [
        pathlib.Path("/root/airflow/dags/COMP315_Group8_project/tfx_airflow_runs/outputs"),
        pathlib.Path.cwd() / "tfx_airflow_runs" / "outputs",
        pathlib.Path.cwd().parent / "tfx_airflow_runs" / "outputs",
    ]
    for r in roots:
        r = r.resolve()
        if r.is_dir():
            return r
    raise FileNotFoundError(
        "No pipeline outputs found. Run the DAG first, or open the notebook from the repo / set path manually."
    )


def latest_version_dir(component_rel: str) -> pathlib.Path:
    \"\"\"e.g. component_rel='StatisticsGen/statistics' -> .../statistics/<max_id>\"\"\"
    base = OUTPUTS / component_rel
    if not base.is_dir():
        raise FileNotFoundError(base)
    ids = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    if not ids:
        raise FileNotFoundError(f"No version dirs under {base}")
    return base / str(max(ids))


OUTPUTS = find_outputs_dir()
print("Using pipeline outputs:", OUTPUTS)"""
    )
)

cells.append(
    new_markdown_cell(
        """## Step 1 — ExampleGen (artifact locations)

Train / eval TFRecord shards produced by `CsvExampleGen` (hash split 2:1)."""
    )
)

cells.append(
    new_code_cell(
        """eg = latest_version_dir("CsvExampleGen/examples")
train_uri = eg / "Split-train"
eval_uri = eg / "Split-eval"
print("ExampleGen version:", eg.name)
print("Train URI:", train_uri)
print("Eval URI:", eval_uri)
for name, u in [("train", train_uri), ("eval", eval_uri)]:
    if u.is_dir():
        n = len(list(u.glob("*")))
        print(f"  {name} shard files (~{n})")"""
    )
)

cells.append(
    new_markdown_cell(
        """## Step 2 — StatisticsGen (TFDV)

Interactive comparison of **train** vs **eval** feature statistics."""
    )
)

cells.append(
    new_code_cell(
        """# StatisticsGen writes binary DatasetFeatureStatisticsList at Split-*/FeatureStats.pb
# (tfdv.load_statistics() expects a TFRecord or text proto file, not a directory.)
stats_root = latest_version_dir("StatisticsGen/statistics")
train_stats = tfdv.load_stats_binary(str(stats_root / "Split-train" / "FeatureStats.pb"))
eval_stats = tfdv.load_stats_binary(str(stats_root / "Split-eval" / "FeatureStats.pb"))
# TFDV 1.x: two protos + names (dict API is only in newer TFDV).
tfdv.visualize_statistics(
    train_stats, eval_stats, lhs_name="train", rhs_name="eval"
)"""
    )
)

cells.append(
    new_markdown_cell(
        """## Step 3 — SchemaGen (TFDV)

Inferred schema from training data."""
    )
)

cells.append(
    new_code_cell(
        """schema_dir = latest_version_dir("SchemaGen/schema")
schema_path = schema_dir / "schema.pbtxt"
schema = tfdv.load_schema_text(str(schema_path))
tfdv.display_schema(schema)"""
    )
)

cells.append(
    new_markdown_cell(
        """## Step 4 — ExampleValidator (TFDV)

Load anomaly artifacts per split (binary `SchemaDiff.pb` → `Anomalies`)."""
    )
)

cells.append(
    new_code_cell(
        """from tensorflow_data_validation.utils.anomalies_util import load_anomalies_binary
from tensorflow_data_validation.utils.display_util import display_anomalies

val_root = latest_version_dir("ExampleValidator/anomalies")
for split in ("Split-train", "Split-eval"):
    p = val_root / split / "SchemaDiff.pb"
    if not p.is_file():
        print(split, ": no SchemaDiff.pb")
        continue
    anom = load_anomalies_binary(str(p))
    print("===", split, "===")
    display_anomalies(anom)"""
    )
)

cells.append(
    new_markdown_cell(
        """## Step 5 — Transform (post-transform statistics)

Distribution of **transformed** features (after TFT `preprocessing_fn`)."""
    )
)

cells.append(
    new_code_cell(
        """post_dir = latest_version_dir("Transform/post_transform_stats")
post_stats = tfdv.load_stats_binary(str(post_dir / "FeatureStats.pb"))
tfdv.visualize_statistics(post_stats)"""
    )
)

cells.append(
    new_markdown_cell(
        """## Step 6 — Trainer (SavedModel)

Keras **SavedModel** from `obesity_tfx/trainer.py`. Servable under **`Format-Serving`**."""
    )
)

cells.append(
    new_code_cell(
        """trainer_run = latest_version_dir("Trainer/model")
saved_model_dir = trainer_run / "Format-Serving"
print("Trainer artifact id:", trainer_run.name)
print("SavedModel (Format-Serving):", saved_model_dir.resolve())
if not saved_model_dir.is_dir():
    print("(Format-Serving not found yet — run the pipeline through Trainer.)")"""
    )
)

cells.append(
    new_markdown_cell(
        """## Step 8 — Evaluator (TFMA)

**Overall metrics**, **time series** (single run), **slices** (`Gender`, `Age_bucket` per pipeline `EvalConfig`), and **plots** when present.

Uses `tensorflow_model_analysis` 0.43 API (`load_eval_result`, `tfma.view.render_*`)."""
    )
)

cells.append(
    new_code_cell(
        """eval_dir = latest_version_dir("Evaluator/evaluation")
print("TFMA evaluation directory:", eval_dir)

eval_result = tfma.load_eval_result(str(eval_dir))
eval_results = tfma.load_eval_results([str(eval_dir)])

# Overall slicing metrics (all slices table)
tfma.view.render_slicing_metrics(eval_result)"""
    )
)

cells.append(
    new_code_cell(
        """# Slice: Gender (as configured in tfx_pipeline EvalConfig)
_spec_gender = ParseDict({"feature_keys": ["Gender"]}, cast(Message, config_pb2.SlicingSpec()))
tfma.view.render_slicing_metrics(eval_result, slicing_spec=_spec_gender)"""
    )
)

cells.append(
    new_code_cell(
        """# Slice: Age_bucket
_spec_age = ParseDict({"feature_keys": ["Age_bucket"]}, cast(Message, config_pb2.SlicingSpec()))
tfma.view.render_slicing_metrics(eval_result, slicing_spec=_spec_age)"""
    )
)

cells.append(
    new_code_cell(
        """# Time series view (useful when multiple eval runs exist; works with one run too)
tfma.view.render_time_series(eval_results)"""
    )
)

cells.append(
    new_code_cell(
        """# Plots exported by TFMA (e.g. metric curves) — skipped if none
try:
    tfma.view.render_plot(eval_result)
except Exception as e:
    print("Plot viewer skipped:", e)"""
    )
)

cells.append(
    new_code_cell(
        """# Summary: how many slice/metric records TFMA materialized
sm = eval_result.slicing_metrics
print("Slicing metric entries:", len(sm) if sm is not None else 0)
if sm:
    first = sm[0]
    print("First entry type:", type(first).__name__)"""
    )
)

cells.append(
    new_markdown_cell(
        """## Step 9 — Pusher (blessed model)

**Pusher** copies the blessed SavedModel to **`serving_model/`** at the project root (`airflow_dags/obesity_dag.py`). Below: ML Metadata path and filesystem `serving_model/`."""
    )
)

cells.append(
    new_code_cell(
        """pushed_run = latest_version_dir("Pusher/pushed_model")
print("Pusher artifact id:", pushed_run.name)
print("Pushed SavedModel root:", pushed_run.resolve())
_serving_root = OUTPUTS.parent.parent / "serving_model"
print("Filesystem serving copy (project serving_model/):", _serving_root.resolve())"""
    )
)

nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3 (tfx-env)", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.7.0"}})

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Wrote", NB_PATH)
