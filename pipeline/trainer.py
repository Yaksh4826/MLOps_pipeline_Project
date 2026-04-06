import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

_LABEL_KEY = 'Obese'

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=tf.data.TFRecordDataset,
        label_key=_LABEL_KEY)
    return dataset

def run_fn(args: FnArgs):
    """Step 6: User-defined training logic for the obesity model."""
    tf_transform_output = tft.TFTransformOutput(args.transform_output)
    
    train_dataset = _input_fn(args.train_files, tf_transform_output)
    eval_dataset = _input_fn(args.eval_files, tf_transform_output)

    # Simple DNN Architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(16,), name='input_features'), # 8 continuous + 8 categorical
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    # TensorBoard callback for requirement compliance
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.model_run_dir)

    model.fit(
        train_dataset,
        steps_per_epoch=args.train_steps,
        validation_data=eval_dataset,
        validation_steps=args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10)

    model.save(args.serving_model_dir, save_format='tf')