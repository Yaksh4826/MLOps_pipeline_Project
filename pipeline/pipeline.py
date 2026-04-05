import os
from tfx import v1 as tfx
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


def create_pipeline(pipeline_name, pipeline_root, data_root, metadata_path):
    # Steps 1-4 (Member A)
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
    # Step 5: Transform (Wired to ExampleGen, SchemaGen, and module file)
    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.join('pipeline', 'transform.py')
        )

    # Step 6: Trainer (Wired to Transform graph/examples and module file)
    trainer = tfx.components.Trainer(
        module_file=os.path.join('pipeline', 'trainer.py'),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=50)
    )

    # Step 7: Resolver (LatestBlessedModelResolver for regression testing)
    model_resolver = resolver.Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path),
        components=[example_gen, statistics_gen, schema_gen, example_validator],
        enable_cache=True,
    )