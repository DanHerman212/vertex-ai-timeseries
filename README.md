# NHITS & GRU Training Pipeline on GCP

This repository contains the end-to-end machine learning pipeline for training and evaluating time-series forecasting models (NHITS and GRU) on Google Cloud Platform (Vertex AI).

## Deployment Checklist

Use this checklist to ensure your environment and project are ready for the first deployment to Vertex AI.

## 1. Prerequisites (Local Environment)
- [ ] **Google Cloud SDK** is installed (`gcloud --version`).
- [ ] **Docker** is installed and running (`docker --version`).
- [ ] **Python 3.11** (or compatible) is installed.
- [ ] **Authenticated** with Google Cloud:
  ```bash
  gcloud auth login
  ```
- [ ] **Project Set**:
  ```bash
  gcloud config set project YOUR_PROJECT_ID
  ```

## 2. One-Time Infrastructure Setup
*Ensure these resources exist in the `us-east1` region.*

- [ ] **Artifact Registry Repository** created:
  ```bash
  gcloud artifacts repositories create ml-pipelines \
      --repository-format=docker \
      --location=us-east1 \
      --description="ML Pipeline Docker Repository"
  ```
- [ ] **GCS Bucket** created:
  ```bash
  gcloud storage buckets create gs://YOUR_UNIQUE_BUCKET_NAME --location=us-east1
  ```
- [ ] **Permissions**:
  - Your user account has `Artifact Registry Writer` (or Owner) role.
  - The Compute Engine default service account has `Storage Object Admin` on the bucket (usually default).

## 3. Project Configuration
- [ ] **Check `deploy_pipeline.sh`**:
  Open the script and verify the variables at the top match your setup:
  - `PROJECT_ID`: Your actual project ID.
  - `REGION`: `us-east1`
  - `BUCKET_NAME`: The bucket you created above.
  - `REPO_NAME`: `ml-pipelines` (or whatever you named it).
  - `BQ_QUERY`: Your specific SQL query for BigQuery.

## 4. Code Verification
- [ ] **Source Code**: Ensure all python scripts (`train_gru.py`, `preprocess.py`, etc.) are inside the `src/` folder.
- [ ] **Dockerfile**: Ensure it contains `COPY src/ .` (we updated this previously).
- [ ] **Pipeline Definition**: Ensure `pipeline.py` points to `src/preprocess.py` and `src/train_gru.py`.

## 5. Deployment
- [ ] **Make Script Executable** (if not already):
  ```bash
  chmod +x deploy_pipeline.sh
  ```
- [ ] **Run Deployment**:
  ```bash
  ./deploy_pipeline.sh
  ```

## 6. Verification
- [ ] **Watch Terminal**: Wait for "Pipeline submitted successfully!".
- [ ] **Vertex AI Console**:
  - Go to [Google Cloud Console > Vertex AI > Pipelines](https://console.cloud.google.com/vertex-ai/pipelines).
  - Click on the run named `gru-training-run-...`.
  - Verify the graph shows: `extract-bq-data` -> `preprocess-component` -> `train-gru-component`.
