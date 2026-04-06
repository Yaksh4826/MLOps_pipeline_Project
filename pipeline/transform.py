import tensorflow as tf
import tensorflow_transform as tft

# Feature names from obesity_dataset_binary.csv
_CONTINUOUS_FEATURES = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
_CATEGORICAL_FEATURES = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'CAEC', 'SCC', 'CALC', 'MTRANS']
_LABEL_KEY = 'Obese'

def _fill_in_missing(x):
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = '' if x.dtype == tf.string else 0
        return tf.squeeze(tf.sparse.to_dense(tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]), default_value), axis=1)
    return x

def preprocessing_fn(inputs):
    """Step 5: User-defined feature engineering logic."""
    outputs = {}

    # Scale continuous features to [0, 1]
    for key in _CONTINUOUS_FEATURES:
        outputs[key] = tft.scale_to_0_1(_fill_in_missing(inputs[key]))

    # Convert categorical features to integer IDs
    for key in _CATEGORICAL_FEATURES:
        outputs[key] = tft.compute_and_apply_vocabulary(_fill_in_missing(inputs[key]))

    # Label remains as is
    outputs[_LABEL_KEY] = _fill_in_missing(inputs[_LABEL_KEY])

    return outputs