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
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TAG}"
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
echo ""
echo "[1/3] Building and Pushing Docker Image..."
# Ensure we are authenticated with the registry
echo "Configuring docker auth for ${REGION}-docker.pkg.dev..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo "Building image..."
docker build --platform linux/amd64 -t $IMAGE_URI .

echo "Pushing image..."
docker push $IMAGE_URI

# 2. Compile Pipeline
echo ""
echo "[2/3] Compiling Pipeline..."
# Ensure kfp is installed locally or use a virtualenv
if ! python -c "import kfp" &> /dev/null; then
    echo "Installing KFP SDK..."
    pip install kfp google-cloud-pipeline-components
fi

python pipeline.py

if [ ! -f "$PIPELINE_JSON" ]; then
    echo "Error: Pipeline compilation failed. $PIPELINE_JSON not found."
    exit 1
fi

# 3. Run Pipeline
echo ""
echo "[3/3] Submitting Pipeline Job to Vertex AI..."
gcloud ai pipelines run \
  --project=$PROJECT_ID \
  --region=$REGION \
  --display-name="gru-training-run-$(date +%Y%m%d-%H%M%S)" \
  --pipeline-file=$PIPELINE_JSON \
  --pipeline-root=$PIPELINE_ROOT \
  --parameter-values="project_id=$PROJECT_ID,bq_query=$BQ_QUERY,bucket_name=$BUCKET_NAME,training_image_uri=$IMAGE_URI"

echo ""
echo "========================================================"
echo "Pipeline submitted successfully!"
echo "Check the Vertex AI Console for progress."
echo "========================================================"
