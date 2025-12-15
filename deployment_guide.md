# NHITS Model Deployment on Google Cloud Platform (GCP)

This guide outlines the workflow to train and deploy the NHITS (Neural Hierarchical Interpolation for Time Series) model using Google Cloud Vertex AI.

## Prerequisites

1.  **GCP Project**: A Google Cloud Project with billing enabled.
2.  **APIs Enabled**: Enable Vertex AI API, Cloud Storage API, and Container Registry API.
3.  **GCS Bucket**: Create a bucket to store model artifacts (e.g., `gs://my-nhits-bucket`).

## Workflow Overview

1.  **Containerize Training Code**: Build a Docker image containing the NHITS code and dependencies.
2.  **Push to Artifact Registry**: Upload the image to GCP.
3.  **Vertex AI Training**: Submit a custom training job using the image.
4.  **Model Registry**: Import the trained model artifact into Vertex AI Model Registry.
5.  **Endpoint Deployment**: Deploy the model to an endpoint for serving predictions.

## Step-by-Step Instructions

### 1. Setup Environment

```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export BUCKET_NAME="your-bucket-name"
export IMAGE_URI="gcr.io/$PROJECT_ID/nhits-training:v1"
```

### 2. Build and Push Docker Image

Navigate to the `nhits_gcp_workflow` directory:

```bash
cd nhits_gcp_workflow
gcloud builds submit --tag $IMAGE_URI .
```

### 3. Run Training Job on Vertex AI

#### Option A: Run Single Custom Job (Manual)

You can submit the job using the `gcloud` CLI:

```bash
gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=nhits-training-job \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=$IMAGE_URI \
  --args="--bucket_name=$BUCKET_NAME"
```

#### Option B: Run Vertex AI Pipeline (Automated)

This workflow includes BigQuery extraction, preprocessing, and training.

1.  **Compile the Pipeline**:
    ```bash
    pip install kfp google-cloud-aiplatform
    python pipeline.py
    ```
    This generates `nhits_pipeline.json`.

2.  **Submit the Pipeline**:
    You can submit this via Python or the Vertex AI Console.
    
    **Using Python:**
    ```python
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.PipelineJob(
        display_name="nhits-pipeline",
        template_path="nhits_pipeline.json",
        parameter_values={
            "project_id": PROJECT_ID,
            "bq_query": "SELECT * FROM `your-project.dataset.table`",
            "bucket_name": BUCKET_NAME,
            "training_image_uri": IMAGE_URI
        },
        pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root"
    )

    job.run()
    ```

### 4. Deploy Model

Once training is complete and the model is saved to GCS:

1.  **Import Model**:
    Go to Vertex AI -> Model Registry -> Import.
    Select the GCS path where `train_model.py` saved the model.
    Use a pre-built container for serving (e.g., PyTorch) or a custom one if `neuralforecast` requires specific dependencies for inference.

2.  **Create Endpoint**:
    Vertex AI -> Endpoints -> Create Endpoint.

3.  **Deploy**:
    Deploy the imported model to the created endpoint.

## About NHITS

NHITS is a deep learning model for time series forecasting that uses hierarchical interpolation. It is known for:
-   **Long-horizon forecasting**: Good at predicting many steps into the future.
-   **Efficiency**: Faster training and inference compared to Transformers.
-   **Accuracy**: State-of-the-art performance on many benchmarks.

We use the `neuralforecast` library by Nixtla in this example, which provides a robust implementation of NHITS.
