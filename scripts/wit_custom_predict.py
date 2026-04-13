# Copyright 2026 COMP315 Group 8 — TensorBoard What-If Tool (web UI) helper.
#
# Used with TensorBoard's custom prediction hook so the What-If Tool runs in the
# browser without Jupyter. Set WIT_SAVEDMODEL_PATH to your Trainer export:
#   .../tfx_airflow_runs/outputs/Trainer/model/<run_id>/Format-Serving
# (or the blessed copy under serving_model/ if you prefer).
#
# Launch (paths are examples — use your absolute paths):
#   export WIT_SAVEDMODEL_PATH=/path/to/Format-Serving
#   tensorboard --logdir=/tmp/wit_tb --bind_all \\
#     --whatif_data_dir=/path/to/dir/containing/tfrecord \\
#     --whatif_use_unsafe_custom_prediction=/path/to/scripts/wit_custom_predict.py
#
# In the WIT setup dialog you still enter placeholder inference host/name if the
# tool requires it; those values are ignored when using this custom function.

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import tensorflow as tf

_infer = None
_input_keys: Optional[List[str]] = None
_feats: Optional[Dict[str, Any]] = None


def _load_model() -> None:
    global _infer, _input_keys, _feats
    if _infer is not None:
        return
    path = os.environ.get("WIT_SAVEDMODEL_PATH")
    if not path or not os.path.isdir(path):
        raise RuntimeError(
            "Set environment variable WIT_SAVEDMODEL_PATH to the SavedModel directory "
            "(Trainer …/Format-Serving, or serving_model)."
        )
    m = tf.saved_model.load(path)
    _infer = m.signatures["serving_default"]
    _input_keys = list(_infer.structured_input_signature[1].keys())
    _feats = {}
    for k in _input_keys:
        spec = _infer.structured_input_signature[1][k]
        _feats[k] = tf.io.FixedLenFeature([1], spec.dtype)


def custom_predict_fn(examples, serving_bundle):
    """TensorBoard WIT entry point: list[tf.train.Example] -> class score lists."""
    del serving_bundle  # Inference dialog fields ignored for local SavedModel.
    _load_model()
    assert _infer is not None and _input_keys is not None and _feats is not None
    serials = [e.SerializeToString() for e in examples]
    parsed = tf.io.parse_example(serials, _feats)
    inp = {k: parsed[k] for k in _input_keys}
    probs = _infer(**inp)["obese_probability"].numpy().reshape(-1)
    return [[float(1 - p), float(p)] for p in probs]
