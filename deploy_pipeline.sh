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

# Load environment variables from .env if present
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Required Variables
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID is not set. Please set it in .env or export it."
    exit 1
fi

if [ -z "$BUCKET_NAME" ]; then
    echo "Error: BUCKET_NAME is not set. Please set it in .env or export it."
    exit 1
fi

REGION=${REGION:-"us-east1"}
REPO_NAME=${REPO_NAME:-"ml-pipelines"}
TENSORFLOW_IMAGE_NAME=${TENSORFLOW_IMAGE_NAME:-"tensorflow-training"}
PYTORCH_IMAGE_NAME=${PYTORCH_IMAGE_NAME:-"pytorch-training"}
PYTORCH_SERVING_IMAGE_NAME=${PYTORCH_SERVING_IMAGE_NAME:-"pytorch-serving"}
# Generate a unique tag based on timestamp if not provided
TAG=${TAG:-"v$(date +%Y%m%d-%H%M%S)"}
BQ_QUERY=${BQ_QUERY:-'select
  arrival_date,
  duration,
  mbt,
  dow
from `mta_historical_v3.ml`'}

# Derived Variables
# Note: We keep the image in us-east1 to avoid re-pushing, but run the pipeline in the configured REGION (us-east1)
if [ -z "$TENSORFLOW_IMAGE_URI" ]; then
    TENSORFLOW_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${TENSORFLOW_IMAGE_NAME}:${TAG}"
    SHOULD_BUILD_TF=true
else
    echo "Using provided TENSORFLOW_IMAGE_URI: $TENSORFLOW_IMAGE_URI"
    SHOULD_BUILD_TF=false
fi

if [ -z "$PYTORCH_IMAGE_URI" ]; then
    PYTORCH_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${PYTORCH_IMAGE_NAME}:${TAG}"
    SHOULD_BUILD_PYTORCH=true
else
    echo "Using provided PYTORCH_IMAGE_URI: $PYTORCH_IMAGE_URI"
    SHOULD_BUILD_PYTORCH=false
fi

if [ -z "$PYTORCH_SERVING_IMAGE_URI" ]; then
    PYTORCH_SERVING_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${PYTORCH_SERVING_IMAGE_NAME}:${TAG}"
    SHOULD_BUILD_SERVING=true
else
    echo "Using provided PYTORCH_SERVING_IMAGE_URI: $PYTORCH_SERVING_IMAGE_URI"
    SHOULD_BUILD_SERVING=false
fi

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
    
    if [ "$SHOULD_BUILD_TF" = true ]; then
        echo "Building TensorFlow Training Image..."
        gcloud builds submit --config <(echo "steps:
- name: 'gcr.io/cloud-builders/docker'
  entrypoint: 'bash'
  args: ['-c', 'docker pull \$_IMAGE_URI || exit 0']
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '--cache-from', '\$_IMAGE_URI', '-t', '\$_IMAGE_URI', '-f', '\$_DOCKERFILE', '.']
images:
- '\$_IMAGE_URI'
substitutions:
  _IMAGE_URI: '$TENSORFLOW_IMAGE_URI'
  _DOCKERFILE: 'docker/Dockerfile'
") .
    else
        echo "Skipping TensorFlow build. Using provided URI: $TENSORFLOW_IMAGE_URI"
    fi
    
    # --------------------------------------------------------------------------
    # Build 2: PyTorch Training Image
    # --------------------------------------------------------------------------
    # Note: 'gcloud builds submit' does not support the '-f' flag to specify a 
    # Dockerfile directly. We must use a Cloud Build configuration (YAML).
    # Below, we use "Process Substitution" <(...) to pass an inline YAML config.
    # We also add a 'docker pull' step to enable caching (--cache-from), which
    # significantly speeds up subsequent builds.
    if [ "$SHOULD_BUILD_PYTORCH" = true ]; then
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
  _DOCKERFILE: 'docker/Dockerfile.nhits'
") .
    else
        echo "Skipping PyTorch build. Using provided URI: $PYTORCH_IMAGE_URI"
    fi

    # --------------------------------------------------------------------------
    # Build 3: PyTorch Serving Image
    # --------------------------------------------------------------------------
    # Same technique as above, but pointing to Dockerfile.serving
    if [ "$SHOULD_BUILD_SERVING" = true ]; then
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
  _DOCKERFILE: 'docker/Dockerfile.serving'
") .
    else
        echo "Skipping PyTorch Serving build. Using provided URI: $PYTORCH_SERVING_IMAGE_URI"
    fi
fi

# 2. Compile Pipeline
echo ""
echo "[2/3] Compiling Pipeline..."
# Only install lightweight compilation dependencies, not the full training requirements
echo "Installing KFP and Pipeline Components..."

# Add user local bin to PATH to ensure installed scripts are found
export PATH="$HOME/.local/bin:$PATH"

# Pinning google-cloud-pipeline-components to a version compatible with Python 3.12
# Using --break-system-packages to bypass PEP 668 restrictions in this environment
pip install -q --break-system-packages "kfp>=2.7.0" "google-cloud-pipeline-components>=2.18.0" "google-cloud-aiplatform>=1.38.0" "google-auth>=2.29.0" "requests>=2.31.0"

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
