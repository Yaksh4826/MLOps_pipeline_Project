import os
from tfx import v1 as tfx

def create_pipeline(pipeline_name, pipeline_root, data_root, metadata_path):
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)
    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs["examples"])
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    )
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )
    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
        components=[example_gen, statistics_gen, schema_gen, example_validator],
        enable_cache=True,
    )