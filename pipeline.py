from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact,
)
import os

# 1. Component: Extract Data from BigQuery
@component(
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow", "db-dtypes"],
    base_image="python:3.11"
)
def extract_bq_data(
    project_id: str,
    query: str,
    output_dataset: Output[Dataset]
):
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    query_job = client.query(query)
    df = query_job.to_dataframe()
    
    # Save to the output artifact path
    # The path provided by KFP usually doesn't have an extension, so we can just write to it
    df.to_csv(output_dataset.path, index=False)

# 2. Component Definition for Custom Scripts
# We define dummy components that define the interface (inputs/outputs)
# and then override the container implementation in the pipeline.

@component(base_image="python:3.11")
def preprocess_component(
    input_csv: Input[Dataset],
    output_csv: Output[Dataset],
):
    pass

@component(base_image="python:3.11")
def train_gru_component(
    input_csv: Input[Dataset],
    bucket_name: str,
    model_dir: Output[Model]
):
    pass

# 3. Pipeline Definition
@dsl.pipeline(
    name="gru-training-pipeline",
    description="Pipeline to extract data, preprocess, and train the GRU model."
)
def gru_pipeline(
    project_id: str,
    bq_query: str,
    bucket_name: str,
    training_image_uri: str
):
    # Step 1: Extract
    # Note: Output artifacts like 'output_dataset' are automatically handled by KFP.
    # We do NOT pass them as arguments.
    extract_task = extract_bq_data(
        project_id=project_id,
        query=bq_query
    )
    
    # Step 2: Preprocess
    # We use the custom image for this step
    preprocess_task = preprocess_component(
        input_csv=extract_task.outputs["output_dataset"]
    )
    preprocess_task.container.set_image(training_image_uri)
    preprocess_task.container.set_entrypoint(["python", "src/preprocess.py"])
    preprocess_task.container.set_args([
        "--input_csv", preprocess_task.inputs["input_csv"].path,
        "--output_csv", preprocess_task.outputs["output_csv"].path
    ])
    
    # Step 3: Train GRU
    train_gru_task = train_gru_component(
        input_csv=preprocess_task.outputs["output_csv"],
        bucket_name=bucket_name
    )
    train_gru_task.container.set_image(training_image_uri)
    train_gru_task.container.set_entrypoint(["python", "src/train_gru.py"])
    # Note: train_gru.py expects --model_dir to be a directory where it saves 'gru_model.keras' and 'scaler.pkl'
    # KFP Output[Model] provides a path (usually a directory or a file path depending on usage).
    # We will pass it as the model_dir.
    train_gru_task.container.set_args([
        "--input_csv", train_gru_task.inputs["input_csv"].path,
        "--bucket_name", bucket_name,
        "--model_dir", train_gru_task.outputs["model_dir"].path
    ])

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=gru_pipeline,
        package_path="gru_pipeline.json"
    )
