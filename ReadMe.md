# COMP315 — MLOps end-to-end pipeline (Group 8)

TFX pipeline for **obesity binary classification**, orchestrated with **Apache Airflow**, with **TFDV** (data validation) and **TFMA** (model evaluation) visualization in **`Visualization.ipynb`**.

## What’s in the repo

| Area | Role |
|------|------|
| `obesity_tfx/` | TFX pipeline (`tfx_pipeline.py`), `Transform`, `Trainer`, Evaluator blessing |
| `airflow_dags/obesity_dag.py` | Airflow DAG entrypoint |
| `data/obesity/` | CSV + dataset notes |
| `Visualization.ipynb` | **Artifact walkthrough**: ExampleGen → … → Trainer → Evaluator → Pusher (see the notebook’s step table) |
| `serving_model/` | **Pusher** writes the blessed **SavedModel** here (TensorFlow Serving / What-If); large files are gitignored except `.gitkeep` |
| `requirements.txt` | Python deps for pipeline + notebooks |

Run the **`obesity_ml_pipeline`** DAG (or local TFX driver) so `tfx_airflow_runs/outputs/` contains component artifacts before opening the notebook.

## TensorBoard — What-If Tool (WIT)

In **`tfx-env`**, install/refresh with:

`pip install 'tensorboard-plugin-wit>=1.8,<2' 'witwidget>=1.8,<2'`

- **`tensorboard-plugin-wit`** — adds the **What-If** tab to TensorBoard (bundled with `tensorboard` from TensorFlow in many setups).
- **`witwidget`** — optional **Jupyter** embedding for the What-If UI.

Start TensorBoard on a log directory (and/or follow [What-If Tool](https://pair-code.github.io/what-if-tool/) docs to attach a **SavedModel** + **examples**). Example:

`tensorboard --logdir /path/to/logs --bind_all`

Then open the UI (default port **6006**) and use the **What-If Tool** plugin tab.
