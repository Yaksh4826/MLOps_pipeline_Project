import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

_LABEL_KEY = "Obese"


def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    if isinstance(file_pattern, list):
        file_pattern = ",".join(file_pattern)

    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        # Transform writes TFRecords with .gz suffix (gzip); must match or reads look "corrupt".
        reader=lambda filenames: tf.data.TFRecordDataset(
            filenames, compression_type="GZIP"
        ),
        label_key=_LABEL_KEY,
    )


def run_fn(args: FnArgs):
    """Step 6: Train Keras model on transformed data; TensorBoard + SavedModel export."""
    tf_transform_output = tft.TFTransformOutput(args.transform_output)

    train_dataset = _input_fn(args.train_files, tf_transform_output)
    eval_dataset = _input_fn(args.eval_files, tf_transform_output)

    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(_LABEL_KEY)

    inputs = {}
    for key in sorted(feature_spec.keys()):
        spec = feature_spec[key]
        dtype = spec.dtype
        inputs[key] = tf.keras.layers.Input(shape=(1,), name=key, dtype=dtype)

    # Concatenate expects a common dtype for the Dense stack; cast vocab / bucket indices to float.
    float_feats = []
    for key in sorted(feature_spec.keys()):
        t = inputs[key]
        if feature_spec[key].dtype != tf.float32:
            t = tf.keras.layers.Lambda(
                lambda x: tf.cast(x, tf.float32),
                name=f"cast_float_{key}",
            )(t)
        float_feats.append(t)

    x = tf.keras.layers.concatenate(float_feats)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    # Named output so TFMA ModelSpec can reference prediction_key if needed.
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="obese_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.model_run_dir)

    train_steps = int(args.train_steps or 100)
    eval_steps = int(args.eval_steps or 50)

    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=train_steps,
        validation_data=eval_dataset,
        validation_steps=eval_steps,
        callbacks=[tensorboard_callback],
    )

    # SavedModel with default serving signature (TensorFlow Serving compatible).
    model.save(args.serving_model_dir, save_format="tf")
