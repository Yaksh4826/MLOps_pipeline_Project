import os 
from tfx import v1 as tfx


#Pipeline configurations

PIPELINE_NAME = "obesity_data_validation_pipeline"

# Adjust this path to your WSL project folder
PIPELINE_ROOT = os.path.join(os.getcwd(), "outputs")
METADATA_PATH = os.path.join(os.getcwd(), "metadata", "metadata.db")
DATA_ROOT = os.path.join(os.getcwd(), "data", "obesity")



# Creating the pipline


def create_pipeline():

    # Step 1 Example Gen
    example_gen = tfx.components.CsvExampleGen(input_base=DATA_ROOT)

    #Step 2 Statistics Gen
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # Step 3: SchemaGen
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True
    )

    # Step 4: ExampleValidator
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"]
    )



    # Building the pipeline and returing it 
        # BUILD THE PIPELINE
    # ---------------------------------------------------------
    return tfx.dsl.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH),
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator
        ]
    )

