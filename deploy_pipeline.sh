#!/bin/bash
set -e

# ==================================================================================
# AUTOMATED DEPLOYMENT SCRIPT FOR VERTEX AI PIPELINE
# ==================================================================================
#
# This script automates the following steps:
# 1. Build & Push: Builds the Docker image (linux/amd64) and pushes it to Artifact Registry.
# 2. Compile: Runs 'pipeline.py' to generate the KFP pipeline specification (JSON).
# 3. Submit: Submits the pipeline job to Vertex AI using 'gcloud ai pipelines run'.
#
# USAGE:
#   ./deploy_pipeline.sh
#
# PREREQUISITES:
#   - gcloud SDK installed and authenticated
#   - Docker installed and running
#   - Python environment with 'kfp' installed
#
# CONFIGURATION:
#   You can override the defaults below by exporting environment variables before running:
#   export PROJECT_ID="my-project"
#   export BUCKET_NAME="my-bucket"
#   export REPO_NAME="my-repo"
#   export BQ_QUERY="SELECT * FROM ..."
#
# ==================================================================================
# Configuration - Override these variables or set them in your environment
# ==================================================================================
PROJECT_ID=${PROJECT_ID:-"time-series-478616"}
REGION=${REGION:-"us-east1"}
REPO_NAME=${REPO_NAME:-"ml-pipelines"}
IMAGE_NAME=${IMAGE_NAME:-"gru-training"}
TAG=${TAG:-"v1"}
BUCKET_NAME=${BUCKET_NAME:-"time-series-478616-ml-pipeline"}
BQ_QUERY=${BQ_QUERY:-'select
  arrival_date,
  duration,
  mbt
from `mta_historical_v3.ml_cleaned`
where extract(year from arrival_date) >= 2024'}

# Derived Variables
# Note: We keep the image in us-east1 to avoid re-pushing, but run the pipeline in the configured REGION (us-east1)
IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"
PIPELINE_ROOT="gs://${BUCKET_NAME}/pipeline_root"
PIPELINE_JSON="gru_pipeline.json"

echo "========================================================"
echo "Starting Deployment for Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Image URI: $IMAGE_URI"
echo "Pipeline Root: $PIPELINE_ROOT"
echo "========================================================"

# Check if required tools are installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed."
    exit 1
fi
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud is not installed."
    exit 1
fi

# 1. Build and Push Docker Image
if [ "$1" == "--skip-build" ]; then
    echo ""
    echo "[1/3] Skipping Docker Build..."
else
    echo ""
    echo "[1/3] Building and Pushing Docker Image..."
    # Use Cloud Build to avoid local disk space issues with large GPU images
    echo "Submitting build to Cloud Build..."
    gcloud builds submit --tag $IMAGE_URI .
fi

# 2. Compile Pipeline
echo ""
echo "[2/3] Compiling Pipeline..."
# Ensure kfp and aiplatform are installed locally
if ! python -c "import kfp; import google.cloud.aiplatform" &> /dev/null; then
    echo "Installing KFP and Vertex AI SDK..."
    pip install kfp google-cloud-pipeline-components google-cloud-aiplatform
fi

python pipeline.py

if [ ! -f "$PIPELINE_JSON" ]; then
    echo "Error: Pipeline compilation failed. $PIPELINE_JSON not found."
    exit 1
fi

# 3. Run Pipeline
echo ""
echo "[3/3] Submitting Pipeline Job to Vertex AI..."
# Use Python script to bypass gcloud CLI version issues
python submit_pipeline.py \
  --project_id="$PROJECT_ID" \
  --region="$REGION" \
  --bucket_name="$BUCKET_NAME" \
  --pipeline_root="$PIPELINE_ROOT" \
  --pipeline_json="$PIPELINE_JSON" \
  --bq_query="$BQ_QUERY" \
  --training_image_uri="$IMAGE_URI"

echo ""
echo "========================================================"
echo "Pipeline submitted successfully!"
echo "Check the Vertex AI Console for progress."
echo "========================================================"
