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
TENSORFLOW_IMAGE_NAME=${TENSORFLOW_IMAGE_NAME:-"tensorflow-training"}
PYTORCH_IMAGE_NAME=${PYTORCH_IMAGE_NAME:-"pytorch-training"}
PYTORCH_SERVING_IMAGE_NAME=${PYTORCH_SERVING_IMAGE_NAME:-"pytorch-serving"}
# Generate a unique tag based on timestamp if not provided
TAG=${TAG:-"v$(date +%Y%m%d-%H%M%S)"}
BUCKET_NAME=${BUCKET_NAME:-"time-series-478616-ml-pipeline"}
BQ_QUERY=${BQ_QUERY:-'select
  arrival_date,
  duration,
  mbt
from `mta_historical_v3.ml_cleaned`
where extract(year from arrival_date) >= 2024'}

# Derived Variables
# Note: We keep the image in us-east1 to avoid re-pushing, but run the pipeline in the configured REGION (us-east1)
TENSORFLOW_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${TENSORFLOW_IMAGE_NAME}:${TAG}"
PYTORCH_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${PYTORCH_IMAGE_NAME}:${TAG}"
PYTORCH_SERVING_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${PYTORCH_SERVING_IMAGE_NAME}:${TAG}"
PIPELINE_ROOT="gs://${BUCKET_NAME}/pipeline_root"
PIPELINE_JSON="forecasting_pipeline.json"

echo "========================================================"
echo "Starting Deployment for Project: $PROJECT_ID"
echo "Region: $REGION"
echo "TensorFlow Image URI: $TENSORFLOW_IMAGE_URI"
echo "PyTorch Image URI: $PYTORCH_IMAGE_URI"
echo "PyTorch Serving Image URI: $PYTORCH_SERVING_IMAGE_URI"
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
    echo "[1/3] Building and Pushing Docker Images..."
    
    echo "Building TensorFlow Training Image..."
    gcloud builds submit --tag $TENSORFLOW_IMAGE_URI .
    
    # --------------------------------------------------------------------------
    # Build 2: PyTorch Training Image
    # --------------------------------------------------------------------------
    # Note: 'gcloud builds submit' does not support the '-f' flag to specify a 
    # Dockerfile directly. We must use a Cloud Build configuration (YAML).
    # Below, we use "Process Substitution" <(...) to pass an inline YAML config.
    # We also add a 'docker pull' step to enable caching (--cache-from), which
    # significantly speeds up subsequent builds.
    echo "Building PyTorch Training Image..."
    gcloud builds submit --config <(echo "steps:
- name: 'gcr.io/cloud-builders/docker'
  entrypoint: 'bash'
  args: ['-c', 'docker pull \$_IMAGE_URI || exit 0']
- name: 'gcr.io/cloud-builders/docker'
  # This step runs: docker build --cache-from \$_IMAGE_URI -t \$_IMAGE_URI -f \$_DOCKERFILE .
  args: ['build', '--cache-from', '\$_IMAGE_URI', '-t', '\$_IMAGE_URI', '-f', '\$_DOCKERFILE', '.']
images:
- '\$_IMAGE_URI'
substitutions:
  _IMAGE_URI: '$PYTORCH_IMAGE_URI'
  _DOCKERFILE: 'Dockerfile.nhits'
") .

    # --------------------------------------------------------------------------
    # Build 3: PyTorch Serving Image
    # --------------------------------------------------------------------------
    # Same technique as above, but pointing to Dockerfile.serving
    echo "Building PyTorch Serving Image..."
    gcloud builds submit --config <(echo "steps:
- name: 'gcr.io/cloud-builders/docker'
  entrypoint: 'bash'
  args: ['-c', 'docker pull \$_IMAGE_URI || exit 0']
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '--cache-from', '\$_IMAGE_URI', '-t', '\$_IMAGE_URI', '-f', '\$_DOCKERFILE', '.']
images:
- '\$_IMAGE_URI'
substitutions:
  _IMAGE_URI: '$PYTORCH_SERVING_IMAGE_URI'
  _DOCKERFILE: 'Dockerfile.serving'
") .
fi

# 2. Compile Pipeline
echo ""
echo "[2/3] Compiling Pipeline..."
# Only install lightweight compilation dependencies, not the full training requirements
echo "Installing KFP and Pipeline Components..."
# Pinning google-cloud-pipeline-components to a version compatible with Python 3.12
pip install -q "kfp>=2.7.0" "google-cloud-pipeline-components>=2.18.0" "google-cloud-aiplatform>=1.38.0" "google-auth>=2.22.0"

# Export the image URI so pipeline.py can use it during compilation
export TENSORFLOW_IMAGE_URI="$TENSORFLOW_IMAGE_URI"
export PYTORCH_IMAGE_URI="$PYTORCH_IMAGE_URI"
export PYTORCH_SERVING_IMAGE_URI="$PYTORCH_SERVING_IMAGE_URI"
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
  --bq_query="$BQ_QUERY"

echo ""
echo "========================================================"
echo "Pipeline submitted successfully!"
echo "Check the Vertex AI Console for progress."
echo "========================================================"
