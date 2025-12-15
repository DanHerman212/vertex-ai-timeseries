from kfp import dsl
from kfp import compiler
from kfp.dsl import (
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
# We define container components that use the custom image directly.

@dsl.container_component
def preprocess_component(
    training_image_uri: str,
    input_csv: dsl.Input[dsl.Dataset],
    output_csv: dsl.Output[dsl.Dataset],
):
    return dsl.ContainerSpec(
        image=training_image_uri,
        command=["python", "src/preprocess.py"],
        args=[
            "--input_csv", input_csv.path,
            "--output_csv", output_csv.path
        ]
    )

@dsl.container_component
def train_gru_component(
    training_image_uri: str,
    input_csv: dsl.Input[dsl.Dataset],
    bucket_name: str,
    model_dir: dsl.Output[dsl.Model]
):
    return dsl.ContainerSpec(
        image=training_image_uri,
        command=["python", "src/train_gru.py"],
        args=[
            "--input_csv", input_csv.path,
            "--bucket_name", bucket_name,
            "--model_dir", model_dir.path
        ]
    )

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
    extract_task = extract_bq_data(
        project_id=project_id,
        query=bq_query
    )
    
    # Step 2: Preprocess
    preprocess_task = preprocess_component(
        training_image_uri=training_image_uri,
        input_csv=extract_task.outputs["output_dataset"]
    )
    
    # Step 3: Train GRU
    train_gru_task = train_gru_component(
        training_image_uri=training_image_uri,
        input_csv=preprocess_task.outputs["output_csv"],
        bucket_name=bucket_name
    )

    # Configure GPU resources
    train_gru_task.set_cpu_limit('4')
    train_gru_task.set_memory_limit('16G')
    train_gru_task.set_gpu_limit(1)
    train_gru_task.add_node_selector_constraint(label_name='cloud.google.com/gke-accelerator', value='nvidia-tesla-t4')

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=gru_pipeline,
        package_path="gru_pipeline.json"
    )
