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
TENSORFLOW_IMAGE_URI = os.environ.get("TENSORFLOW_IMAGE_URI")
PYTORCH_IMAGE_URI = os.environ.get("PYTORCH_IMAGE_URI")
PYTORCH_SERVING_IMAGE_URI = os.environ.get("PYTORCH_SERVING_IMAGE_URI")

if not TENSORFLOW_IMAGE_URI or not PYTORCH_IMAGE_URI:
    raise ValueError("Image URIs must be set via environment variables (TENSORFLOW_IMAGE_URI, PYTORCH_IMAGE_URI)")

# 1. Component: Extract Data from BigQuery
@dsl.container_component
def extract_bq_data(
    project_id: str,
    query: str,
    output_dataset: dsl.Output[dsl.Dataset]
):
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
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
        image=TENSORFLOW_IMAGE_URI,
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
    test_dataset: dsl.Output[dsl.Dataset],
):
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "src/train_gru.py"],
        args=[
            "--input_csv", input_csv.path,
            "--model_dir", model_dir.path,
            "--test_dataset_path", test_dataset.path
        ]
    )

@dsl.container_component
def evaluate_gru_component(
    test_dataset: dsl.Input[dsl.Dataset],
    model_dir: dsl.Input[artifact_types.UnmanagedContainerModel],
    input_csv: dsl.Input[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics],
    loss_plot: dsl.Output[dsl.HTML],
    prediction_plot: dsl.Output[dsl.HTML],
):
    return dsl.ContainerSpec(
        image=TENSORFLOW_IMAGE_URI,
        command=["python", "src/evaluate_gru.py"],
        args=[
            "--test_dataset_path", test_dataset.path,
            "--model_dir", model_dir.path,
            "--input_csv", input_csv.path,
            "--metrics_output_path", metrics.path,
            "--plot_output_path", loss_plot.path,
            "--prediction_plot_path", prediction_plot.path
        ]
    )

@dsl.container_component
def train_nhits_component(
    input_csv: dsl.Input[dsl.Dataset],
    model_dir: dsl.Output[artifact_types.UnmanagedContainerModel],
    df_output_csv: dsl.Output[dsl.Dataset],
    logs_dir: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=PYTORCH_IMAGE_URI,
        command=["python", "src/train_nhits.py"],
        args=[
            "--input_csv", input_csv.path,
            "--model_dir", model_dir.path,
            "--df_output_csv", df_output_csv.path,
            "--logs_dir", logs_dir.path
        ]
    )

@dsl.container_component
def evaluate_nhits_component(
    df_csv: dsl.Input[dsl.Dataset],
    model_dir: dsl.Input[artifact_types.UnmanagedContainerModel],
    logs_dir: dsl.Input[dsl.Artifact],
    metrics: dsl.Output[dsl.Metrics],
    html_summary: dsl.Output[dsl.HTML],
):
    return dsl.ContainerSpec(
        image=PYTORCH_IMAGE_URI,
        command=["/bin/bash", "-c"],
        args=[
            """
            echo "DEBUG: Starting evaluation component"
            echo "DEBUG: Current directory: $(pwd)"
            echo "DEBUG: Listing src directory:"
            ls -R src/
            echo "DEBUG: Python version:"
            python --version
            echo "DEBUG: Running script..."
            python -u src/evaluate_nhits.py \
            --df_csv_path "$0" \
            --model_dir "$1" \
            --logs_dir "$2" \
            --metrics_output_path "$3" \
            --html_output_path "$4"
            """,
            df_csv.path,
            model_dir.path,
            logs_dir.path,
            metrics.path,
            html_summary.path
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
    name="forecasting-training-pipeline",
    description="Pipeline to extract data, preprocess, and train both GRU and N-HiTS models."
)
def forecasting_pipeline(
    project_id: str,
    bq_query: str,
    region: str = "us-east1",
    model_display_name: str = "gru-model-v1",
    nhits_model_display_name: str = "nhits-model-v1"
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
    # Disable caching to ensure we pick up the latest code changes in the image
    preprocess_task.set_caching_options(False)
    
    # Step 3: Train GRU
    train_gru_task = train_gru_component(
        input_csv=preprocess_task.outputs["output_csv"]
    )

    # Configure GPU resources
    train_gru_task.set_cpu_limit('4')
    train_gru_task.set_memory_limit('16G')
    train_gru_task.set_gpu_limit(1)
    train_gru_task.set_accelerator_type('NVIDIA_TESLA_T4')
    
    # Step 3b: Train N-HiTS
    train_nhits_task = train_nhits_component(
        input_csv=preprocess_task.outputs["output_csv"]
    )
    train_nhits_task.set_cpu_limit('8')
    train_nhits_task.set_memory_limit('32G')
    train_nhits_task.set_gpu_limit(1)
    train_nhits_task.set_accelerator_type('NVIDIA_TESLA_T4')

    # Step 3.5: Attach Serving Spec (GRU)
    # We attach the serving container image URI to the model artifact metadata
    # so that Vertex AI knows which image to use for deployment.
    model_with_metadata_task = attach_serving_spec(
        original_model=train_gru_task.outputs["model_dir"],
        serving_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"
    )

    # Step 4: Upload to Model Registry (GRU)
    model_upload_task = ModelUploadOp(
        project=project_id,
        location=region,
        display_name=model_display_name,
        unmanaged_container_model=model_with_metadata_task.outputs["model_with_spec"],
    )

    # Step 3.5b: Attach Serving Spec (N-HiTS)
    nhits_model_with_metadata_task = attach_serving_spec(
        original_model=train_nhits_task.outputs["model_dir"],
        serving_image_uri=PYTORCH_SERVING_IMAGE_URI
    )

    # Step 4b: Upload to Model Registry (N-HiTS)
    nhits_model_upload_task = ModelUploadOp(
        project=project_id,
        location=region,
        display_name=nhits_model_display_name,
        unmanaged_container_model=nhits_model_with_metadata_task.outputs["model_with_spec"],
    )

    # Step 5: Evaluate GRU
    evaluate_gru_task = evaluate_gru_component(
        test_dataset=train_gru_task.outputs["test_dataset"],
        model_dir=train_gru_task.outputs["model_dir"],
        input_csv=preprocess_task.outputs["output_csv"]
    )
    # Assign GPU to evaluation task to support CudnnRNNV3 ops
    evaluate_gru_task.set_cpu_limit('4')
    evaluate_gru_task.set_memory_limit('16G')
    evaluate_gru_task.set_gpu_limit(1)
    evaluate_gru_task.set_accelerator_type('NVIDIA_TESLA_T4')
    
    # Step 6: Evaluate N-HiTS
    evaluate_nhits_task = evaluate_nhits_component(
        df_csv=train_nhits_task.outputs["df_output_csv"],
        model_dir=train_nhits_task.outputs["model_dir"],
        logs_dir=train_nhits_task.outputs["logs_dir"]
    )
    evaluate_nhits_task.set_cpu_limit('8')
    evaluate_nhits_task.set_memory_limit('32G')
    evaluate_nhits_task.set_gpu_limit(1)
    evaluate_nhits_task.set_accelerator_type('NVIDIA_TESLA_T4')

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=forecasting_pipeline,
        package_path="forecasting_pipeline.json"
    )
