# Operations Guide: N-HiTS Forecasting Pipeline

This guide details the operational procedures for training, testing, and deploying the N-HiTS forecasting system.

## 1. Prerequisites & Setup

Ensure your environment is configured correctly before running any commands.

### Environment Variables
Create a `.env` file in the root directory (copy from `.env.example`):
```bash
PROJECT_ID=your-project-id
REGION=us-east1
BUCKET_NAME=your-bucket-name
REPO_NAME=ml-pipelines
WEATHER_API_KEY=your-api-key # uses visualcrossing.com
# Streaming Config
ZONE=us-east1-b
INSTANCE_NAME=streaming-test-instance
ENDPOINT_ID=your-vertex-endpoint-id
TOPIC_ID=vehicle-position-updates
SUBSCRIPTION_ID=streaming-sub
```

### Authentication
Ensure you are authenticated with Google Cloud:
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project $PROJECT_ID
```

---

## 2. Model Training & Deployment (Vertex AI)

This step launches Vertex AI Pipelines, an MLOPs solution to track experiments, metadata and model artifacts.

### Run the Pipeline
```bash
make deploy
```
*   **What it does**: Builds Docker images, compiles the Kubeflow pipeline, and submits a job to Vertex AI Pipelines.
*   **Duration**: ~15-20 minutes (depending on training epochs).

### Deploy to Endpoint (Manual Step)
Once the pipeline completes successfully:
1.  Go to **Vertex AI Model Registry**.
2.  Select `nhits-model-v1`.
3.  Click on the new version created by the pipeline.
4.  Click **Deploy to Endpoint**.
    *   **Endpoint**: Create new or select existing.
    *   **Machine Type**: `n1-standard-2` (CPU) is sufficient.
    *   **Traffic Split**: 100%.
5.  **Important**: Copy the new **Endpoint ID** and update `ENDPOINT_ID` in your `.env` file.

---

## 3. Testing Serving Logic (Local)

Before relying on the production endpoint, verify the serving container logic locally. This tests the `src/serve.py` logic (timestamp alignment, CPU/GPU fallback, etc.).

### Run Local Test
```bash
./scripts/test_serving_local.sh
```
*   **What it does**: Builds the serving image locally, runs it in a Docker container, performs a health check, and sends a sample prediction request.
*   **Success Criteria**:
    *   Health Check: `{"status": "healthy"}`
    *   Prediction: JSON response with `NHITS-median` values.

---

## 4. Integration Testing (GCE + DirectRunner)

This step runs the full pipeline on a single GCE VM using the `DirectRunner`. This is an intermediate step to verify the end-to-end flow (Ingestion -> Pub/Sub -> Pipeline -> Vertex AI -> Firestore) before scaling out to Dataflow.

### Run Integration Test
```bash
make setup-gce
```
*   **What it does**:
    1.  Provisions a GCE instance (`streaming-test-instance`).
    2.  Deploys the Ingestion Service as a background daemon.
    3.  Runs the Streaming Pipeline in the foreground (DirectRunner).
*   **Verification**:
    *   Watch the terminal output for:
        *   `🚆 ARRIVAL @ ORIGIN`
        *   `🏁 ARRIVAL @ TARGET`
        *   `🔮 PREDICTION RECEIVED`
*   **Stopping**: Press `Ctrl+C` to stop the pipeline. The ingestion service will keep running in the background. You can use the `make stop` command to stop the GCE instance from fetching messages. 

---

## 5. Production Streaming (End-to-End)

The production system consists of two components: an **Ingestion Service** (Producer) running on GCE, and a **Streaming Pipeline** (Consumer) running on Dataflow.

### A. Deploy Ingestion Service
This spins up a GCE VM to pull live MTA data and publish it to Pub/Sub.
```bash
make deploy-ingestion
```
*   **Verification**:
    *   Check the VM status: `gcloud compute instances list`
    *   Check Pub/Sub topic: Ensure messages are being published to `vehicle-position-updates`.

### B. Deploy Streaming Pipeline
This submits the Apache Beam pipeline to Google Cloud Dataflow.
```bash
make deploy-streaming
```
*   **Verification**:
    *   Go to **Dataflow Console**.
    *   Look for job `nhits-streaming-[timestamp]`.
    *   Ensure the job state becomes **Running**.

---

## 6. Monitoring & Validation

### Vertex AI Endpoint
*   **Console**: Vertex AI > Endpoints.
*   **Metrics**: Check **Prediction count** and **Latency**.
*   **Logs**: Check for `200 OK` on `/predict` calls.

### Dataflow
*   **Console**: Dataflow > Jobs.
*   **Metrics**: Check **System Lag** and **Data Freshness**.
*   **Logs**: Look for "🔮 PREDICTION RECEIVED" in the worker logs.

### Firestore (Results)
*   **Console**: Firestore > Data.
*   **Collection**: `predictions`.
*   **Validation**: Ensure new documents are appearing with recent timestamps.

---

## 7. Teardown & Cleanup

To stop costs, remove resources when not in use.

### Automated Teardown
Removes the GCE instance, Pub/Sub resources, Vertex AI Endpoint, and cancels active Dataflow jobs.
```bash
make teardown
```

### Manual Cleanup (Optional)
1.  **Firestore**: The `predictions` collection is not deleted automatically. Delete it from the Firebase Console if desired.
2.  **Container Registry**: Old Docker images in Artifact Registry are not deleted.
2.  **Undeploy Endpoint**: Go to Vertex AI Endpoints and undeploy the model to stop the node billing.
