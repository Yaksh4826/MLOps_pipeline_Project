"""Obesity ML pipeline in TFX, run from Apache Airflow."""

import logging
import os

import importlib

import tensorflow_model_analysis as tfma

# MetricConfig: public export (TFMA ≥0.30) or generated proto (older / minimal installs).
try:
    from tensorflow_model_analysis import MetricConfig
except ImportError:  # pragma: no cover - version-dependent
    _cfg = importlib.import_module("tensorflow_model_analysis.proto.config_pb2")
    MetricConfig = _cfg.MetricConfig

from tfx import v1 as tfx
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2

logger = logging.getLogger(__name__)

_LABEL_KEY = "Obese"
# CsvExampleGen defaults to every file under input_base; README/other files cause
# "Files in same split ... have different header" — ingest only the dataset CSV.
_DEFAULT_INPUT_CSV = "obesity_dataset_binary.csv"


def _build_eval_config():
    """Step 8: TFMA — slices + accuracy/AUC; one model (no baseline resolver required for first green run)."""
    return tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key=_LABEL_KEY)],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=["Gender"]),
            tfma.SlicingSpec(feature_keys=["Age_bucket"]),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    MetricConfig(class_name="BinaryAccuracy"),
                    MetricConfig(class_name="AUC"),
                ],
                thresholds={
                    "binary_accuracy": tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.45}
                        ),
                    )
                },
            )
        ],
    )


def create_pipeline(
    pipeline_name,
    pipeline_root,
    data_root,
    metadata_path,
    serving_model_dir,
    input_csv_pattern=None,
):
    """Steps 1–10: full TFX pipeline through Pusher; ML Metadata at metadata_path."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(serving_model_dir, exist_ok=True)

    csv_pattern = input_csv_pattern or _DEFAULT_INPUT_CSV

    # Step 1: ExampleGen — 2:1 train:eval split (hash buckets), TFRecords under pipeline_root.
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=1),
            ]
        )
    )
    input_config = example_gen_pb2.Input(
        splits=[
            # name must be non-empty (example_gen.proto); pattern is relative to input_base.
            example_gen_pb2.Input.Split(name="obesity_csv", pattern=csv_pattern),
        ]
    )
    example_gen = tfx.components.CsvExampleGen(
        input_base=data_root,
        input_config=input_config,
        output_config=output_config,
    )

    # Step 2: StatisticsGen
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # Step 3: SchemaGen
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    )

    # Step 4: ExampleValidator
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    # Step 5: Transform
    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.join(base_dir, "transform.py"),
    )

    # Step 6: Trainer — train/eval step counts from component config
    trainer = tfx.components.Trainer(
        module_file=os.path.join(base_dir, "trainer.py"),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=50),
    )

    eval_config = _build_eval_config()

    # Step 8: Evaluator (TFMA) — transformed eval split matches SavedModel; no baseline on first runs.
    evaluator = tfx.components.Evaluator(
        examples=transform.outputs["transformed_examples"],
        model=trainer.outputs["model"],
        eval_config=eval_config,
    )

    # Step 9: Pusher — deploy blessed model only
    pusher = tfx.components.Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    _log_pipeline_artifacts(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        data_root=data_root,
        metadata_path=metadata_path,
        serving_model_dir=serving_model_dir,
        example_gen=example_gen,
        statistics_gen=statistics_gen,
        schema_gen=schema_gen,
        example_validator=example_validator,
        evaluator=evaluator,
        pusher=pusher,
    )

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            evaluator,
            pusher,
        ],
        enable_cache=False,
    )


def _channel_uri(channel):
    """Best-effort URI for logs; some TFX versions only materialize this after a run."""
    u = getattr(channel, "uri", None) or getattr(channel, "_uri", None)
    return u if u else "(populated under pipeline_root when the component runs)"


def _log_pipeline_artifacts(
    pipeline_name,
    pipeline_root,
    data_root,
    metadata_path,
    serving_model_dir,
    example_gen,
    statistics_gen,
    schema_gen,
    example_validator,
    evaluator,
    pusher,
):
    """Emit paths so Airflow / local logs show where splits, stats, schema, anomalies, eval, push live."""
    eg_uri = _channel_uri(example_gen.outputs["examples"])
    stats_uri = _channel_uri(statistics_gen.outputs["statistics"])
    schema_uri = _channel_uri(schema_gen.outputs["schema"])
    anomalies_uri = _channel_uri(example_validator.outputs["anomalies"])
    eval_uri = _channel_uri(evaluator.outputs["evaluation"])
    pushed_uri = _channel_uri(pusher.outputs["pushed_model"])

    lines = [
        "",
        "=" * 72,
        f"TFX pipeline: {pipeline_name}",
        f"  ML Metadata DB: {metadata_path}",
        f"  CSV input base: {data_root}",
        f"  Pipeline root:  {pipeline_root}",
        f"  Serving dir:    {serving_model_dir}",
        "  Train:eval split ratio: 2:1 (hash buckets: train=2, eval=1)",
        "",
        "Step 1 — ExampleGen examples channel URI:",
        f"  {eg_uri}",
        "",
        "Step 2 — StatisticsGen statistics URI:",
        f"  {stats_uri}",
        "",
        "Step 3 — SchemaGen schema URI:",
        f"  {schema_uri}",
        "",
        "Step 4 — ExampleValidator anomalies URI (check after each run):",
        f"  {anomalies_uri}",
        "",
        "Step 8 — Evaluator (TFMA) evaluation URI:",
        f"  {eval_uri}",
        "",
        "Step 9 — Pusher pushed_model URI:",
        f"  {pushed_uri}",
        "",
        "On disk, artifacts typically appear under:",
        f"  {pipeline_root}/<pipeline_run_id>/CsvExampleGen|StatisticsGen|...|Evaluator|Pusher|...",
        "=" * 72,
        "",
    ]
    text = "\n".join(lines)
    print(text)
    logger.info(text)
