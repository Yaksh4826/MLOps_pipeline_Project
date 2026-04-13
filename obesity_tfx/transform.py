import tensorflow as tf
import tensorflow_transform as tft

# Feature names from obesity_dataset_binary.csv
_CONTINUOUS_FEATURES = [
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
]
_CATEGORICAL_FEATURES = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "SMOKE",
    "CAEC",
    "SCC",
    "CALC",
    "MTRANS",
]
_LABEL_KEY = "Obese"


def _fill_in_missing(x):
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = "" if x.dtype == tf.string else 0
        dense = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]), default_value
        )
        return tf.squeeze(dense, axis=1)
    return x


def preprocessing_fn(inputs):
    """Step 5: TFT transforms — z-score numerics, vocabulary for categoricals, float label."""
    outputs = {}

    for key in _CONTINUOUS_FEATURES:
        x = _fill_in_missing(inputs[key])
        outputs[key] = tft.scale_to_z_score(x)

    for key in _CATEGORICAL_FEATURES:
        outputs[key] = tft.compute_and_apply_vocabulary(_fill_in_missing(inputs[key]))

    # Fixed age bands (years) for slicing / interpretability — tft.apply_buckets (rank-2 boundaries).
    age = _fill_in_missing(inputs["Age"])
    age = tf.cast(age, tf.float32)
    boundaries = tf.constant(
        [[18.0, 25.0, 35.0, 45.0, 55.0, 65.0]], dtype=tf.float32
    )
    outputs["Age_bucket"] = tft.apply_buckets(age, boundaries)

    y = tf.cast(_fill_in_missing(inputs[_LABEL_KEY]), tf.float32)
    outputs[_LABEL_KEY] = tf.reshape(y, [-1, 1])

    return outputs
