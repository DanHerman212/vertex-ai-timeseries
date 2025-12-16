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
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp
import os

# Get image URI from environment variable (injected by deploy script)
TRAINING_IMAGE_URI = os.environ.get("TRAINING_IMAGE_URI", "us-east1-docker.pkg.dev/time-series-478616/ml-pipelines/gru-training:v1")

# 1. Component: Extract Data from BigQuery
@dsl.container_component
def extract_bq_data(
    project_id: str,
    query: str,
    output_dataset: dsl.Output[dsl.Dataset]
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        command=["python", "src/extract.py"],
        args=[
            "--project_id", project_id,
            "--query", query,
            "--output_csv", output_dataset.path
        ]
    )

# 2. Component Definition for Custom Scripts
# We define container components that use the custom image directly.

@dsl.container_component
def preprocess_component(
    input_csv: dsl.Input[dsl.Dataset],
    output_csv: dsl.Output[dsl.Dataset],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        command=["python", "src/preprocess.py"],
        args=[
            "--input_csv", input_csv.path,
            "--output_csv", output_csv.path
        ]
    )

@dsl.container_component
def train_gru_component(
    input_csv: dsl.Input[dsl.Dataset],
    model_dir: dsl.Output[artifact_types.UnmanagedContainerModel],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        command=["python", "src/train_gru.py"],
        args=[
            "--input_csv", input_csv.path,
            "--model_dir", model_dir.path
        ]
    )

@dsl.container_component
def evaluate_model_component(
    input_csv: dsl.Input[dsl.Dataset],
    model_dir: dsl.Input[artifact_types.UnmanagedContainerModel],
    metrics: dsl.Output[dsl.Metrics],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        command=["python", "src/evaluate_models.py"],
        args=[
            "--input_csv", input_csv.path,
            "--model_dir", model_dir.path,
            "--metrics_output_path", metrics.path
        ]
    )

# Helper component to attach serving metadata
@dsl.component(base_image="python:3.10", packages_to_install=["google-cloud-pipeline-components"])
def attach_serving_spec(
    original_model: dsl.Input[artifact_types.UnmanagedContainerModel],
    model_with_spec: dsl.Output[artifact_types.UnmanagedContainerModel],
    serving_image_uri: str
):
    model_with_spec.uri = original_model.uri
    model_with_spec.metadata = {
        "containerSpec": {
            "imageUri": serving_image_uri
        }
    }

# 3. Pipeline Definition
@dsl.pipeline(
    name="gru-training-pipeline",
    description="Pipeline to extract data, preprocess, and train the GRU model."
)
def gru_pipeline(
    project_id: str,
    bq_query: str,
    region: str = "us-east1",
    model_display_name: str = "gru-model-v1"
):
    # Step 1: Extract
    extract_task = extract_bq_data(
        project_id=project_id,
        query=bq_query
    )
    
    # Step 2: Preprocess
    preprocess_task = preprocess_component(
        input_csv=extract_task.outputs["output_dataset"]
    )
    
    # Step 3: Train GRU
    train_gru_task = train_gru_component(
        input_csv=preprocess_task.outputs["output_csv"]
    )

    # Configure GPU resources
    train_gru_task.set_cpu_limit('4')
    train_gru_task.set_memory_limit('16G')
    train_gru_task.set_gpu_limit(1)
    train_gru_task.set_accelerator_type('NVIDIA_TESLA_T4')
    
    # Fallback to CPU for now due to Quota issues
    # train_gru_task.set_cpu_limit('8')
    # train_gru_task.set_memory_limit('32G')

    # Step 3.5: Attach Serving Spec
    # We attach the serving container image URI to the model artifact metadata
    # so that Vertex AI knows which image to use for deployment.
    model_with_metadata_task = attach_serving_spec(
        original_model=train_gru_task.outputs["model_dir"],
        serving_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-17:latest"
    )

    # Step 4: Upload to Model Registry
    model_upload_task = ModelUploadOp(
        project=project_id,
        location=region,
        display_name=model_display_name,
        unmanaged_container_model=model_with_metadata_task.outputs["model_with_spec"],
    )


    # Step 5: Evaluate
    evaluate_task = evaluate_model_component(
        input_csv=preprocess_task.outputs["output_csv"],
        model_dir=train_gru_task.outputs["model_dir"]
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=gru_pipeline,
        package_path="gru_pipeline.json"
    )
