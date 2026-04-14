# COMP315 — MLOps End-to-End Pipeline (Group 8)

A TFX pipeline for **obesity binary classification**, orchestrated with **Apache Airflow**. Data validation uses **TFDV**, model evaluation uses **TFMA**, and interactive model exploration uses the **TensorBoard What-If Tool**. Artifact visualizations are collected in **`Visualization.ipynb`**.

## Repository layout

| Directory / file | Description |
|---|---|
| `obesity_tfx/` | TFX pipeline definition (`tfx_pipeline.py`), feature engineering (`transform.py`), and model training (`trainer.py`) |
| `airflow_dags/obesity_dag.py` | Airflow DAG that orchestrates the pipeline end-to-end |
| `data/obesity/` | Source CSV and dataset documentation |
| `Visualization.ipynb` | Step-by-step artifact walkthrough: ExampleGen → StatisticsGen → SchemaGen → ExampleValidator → Transform → Trainer → Evaluator → Pusher |
| `serving_model/` | Pusher writes the blessed SavedModel here (gitignored except `.gitkeep`) |
| `scripts/` | Helper scripts for the What-If Tool and notebook generation |
| `requirements.txt` | Python dependencies for the pipeline and notebooks |

## Running the pipeline

The **`obesity_ml_pipeline`** DAG (or the local driver `python run_local.py`) produces all component artifacts under `tfx_airflow_runs/outputs/` (Airflow) or `tfx_pipeline_runs/` (local). The pipeline must complete at least once before opening the notebook or the What-If dashboard.

---

## What-If Tool dashboard

The What-If Tool (WIT) is a TensorBoard plugin that provides interactive model exploration — counterfactual analysis, partial dependence plots, and fairness metrics — all in the browser, no code required.

### Prerequisites

- A completed pipeline run (so SavedModel and transformed eval TFRecords exist on disk).
- The `tfx-env` conda environment with `tensorboard-plugin-wit` installed:

```bash
conda activate tfx-env
pip install 'tensorboard-plugin-wit>=1.8,<2'
```

### Quick start (one command)

From the repository root (`~/COMP315_Group8_project`):

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate tfx-env && bash run_wit_ui.sh
```

This automatically:
1. Syncs `scripts/` into the Airflow project copy.
2. Finds the latest Pusher SavedModel and Transform eval TFRecords.
3. Copies the eval file next to the model (TensorBoard security requirement).
4. Launches TensorBoard with the What-If plugin and custom prediction enabled.

The terminal prints the paths it used. Open **`http://localhost:6006`** in a browser.

### What-If setup dialog

Once TensorBoard is running, select **What-If Tool** from the top-right plugin menu and fill in the setup dialog:

| Field | Value |
|---|---|
| Path to examples | The `wit_eval.tfrecord.gz` path printed in the terminal (under the Pusher model directory) |
| Model type | Classification (binary) |
| Inference | Custom prediction function (loaded automatically from the launch command) |

Inference address and model name fields can be left as placeholders — they are ignored when using the custom prediction script.

### Manual launch (if the quick-start script is not available)

```bash
conda activate tfx-env

export WIT_SAVEDMODEL_PATH='/root/airflow/dags/COMP315_Group8_project/tfx_airflow_runs/outputs/Pusher/pushed_model/34'

cp /root/airflow/dags/COMP315_Group8_project/tfx_airflow_runs/outputs/Transform/transformed_examples/46/Split-eval/transformed_examples-00000-of-00001.gz \
   "$WIT_SAVEDMODEL_PATH/wit_eval.tfrecord.gz"

cd ~/COMP315_Group8_project

tensorboard --logdir="$WIT_SAVEDMODEL_PATH" --bind_all \
  --whatif-data-dir="$WIT_SAVEDMODEL_PATH" \
  --whatif-use-unsafe-custom-prediction="$(pwd)/scripts/wit_custom_predict.py"
```

Replace `pushed_model/34` and `transformed_examples/46` with the artifact IDs from the pipeline run being explored. Flags use **hyphens** (`--whatif-data-dir`), not underscores.

### Dashboard tabs

| Tab | What it shows |
|---|---|
| **Datapoint editor** | Individual examples with real-time score updates when features are edited. Nearest-counterfactual mode highlights the smallest feature diff that changes the prediction. |
| **Performance & Fairness** | Accuracy, false-positive rate, false-negative rate, and F1 across slices (e.g. Gender, Age_bucket). A threshold slider shows how metrics shift as the decision boundary moves. |
| **Features** | Feature distributions (histograms) and partial-dependence plots showing how predictions change as a single feature varies. |

---

## References

- [What-If Tool documentation](https://pair-code.github.io/what-if-tool/)
- [TensorBoard WIT tutorial](https://pair-code.github.io/what-if-tool/learn/tutorials/tensorboard/)
- Hapke & Nelson, *Building Machine Learning Pipelines*, O'Reilly, 2020 (Chapters 7 & 8)
