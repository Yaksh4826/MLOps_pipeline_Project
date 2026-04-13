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

## TensorBoard — What-If Tool (WIT) in the **web browser**

Use the **What-If** UI inside **TensorBoard** (not Jupyter): open `http://localhost:6006`, use the top-right menu to switch to **What-If**, then complete the setup dialog. Official overview: [What-If Tool — TensorBoard](https://pair-code.github.io/what-if-tool/learn/tutorials/tensorboard/).

### 1. Install the plugin

In **`tfx-env`**:

`pip install 'tensorboard-plugin-wit>=1.8,<2'`

(`witwidget` is only needed if you embed WIT inside a notebook; you can skip it for the web UI.)

### 2. Point the tool at **eval TFRecords** (transformed schema)

WIT reads **`tf.Example`** TFRecords from disk. Use the **same transformed eval split** as TFMA:

- `tfx_airflow_runs/outputs/Transform/transformed_examples/<id>/Split-eval/*.gz`

Optional: merge shards into one file so the file picker is simple:

```bash
python scripts/prep_wit_eval_tfrecord.py \
  --pattern '/root/airflow/dags/COMP315_Group8_project/tfx_airflow_runs/outputs/Transform/transformed_examples/<id>/Split-eval/*.gz' \
  --max-examples 500 \
  --out /tmp/wit_eval.tfrecord.gz
```

Put that file in a directory you pass to TensorBoard as the **what-if data directory** (see below), e.g. `/tmp`.

### 3. Load your **SavedModel** (this project: Keras `serving_default`)

**Option A — Custom prediction (simplest for this repo’s multi-input Keras model)**

This uses `scripts/wit_custom_predict.py` so you do **not** need TensorFlow Serving.

```bash
export WIT_SAVEDMODEL_PATH='/root/airflow/dags/COMP315_Group8_project/tfx_airflow_runs/outputs/Trainer/model/<run_id>/Format-Serving'
tensorboard --logdir=/tmp/wit_tb --bind_all \
  --whatif_data_dir=/tmp \
  --whatif_use_unsafe_custom_prediction="$(pwd)/scripts/wit_custom_predict.py"
```

If your TensorBoard build uses **hyphenated** flags instead, try: `--whatif-data-dir` and `--whatif-use-unsafe-custom-prediction`. Check with `tensorboard --helpfull | grep -i whatif`.

Then in the WIT **setup** dialog:

- **Path to examples:** `/tmp/wit_eval.tfrecord.gz` (or a shard path under `--whatif_data_dir`).
- **Model type:** classification (binary).
- Choose **custom prediction function** / follow prompts so TensorBoard uses the script above (wording varies by version).
- Class names (optional): add a small text file with two lines, `Not obese` then `Obese`, and point the dialog at it if offered.

**Option B — TensorFlow Serving**

Serve the same **Format-Serving** directory with TensorFlow Serving’s **Predict** API, then in the WIT dialog set **inference address** `host:port`, **model name**, enable **uses Predict API**, and wire input/output tensor names to match your `serving_default` signature. Dataset path rules are the same; see the [tutorial](https://pair-code.github.io/what-if-tool/learn/tutorials/tensorboard/).

### 4. Course screenshots (in the web UI)

- **Counterfactuals:** explore datapoints → **Counterfactuals**; capture **≥3** examples and write **one paragraph each**.
- **Fairness / threshold:** **Performance & Fairness** → set a **threshold**; slice by **`Gender`** (and optionally **`Age_bucket`**); **one** screenshot + paragraph.
- **Distributions + partial dependence:** **Features** tab → histograms and **partial dependence** plots; short interpretation for your report.
