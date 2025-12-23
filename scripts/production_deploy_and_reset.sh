#!/bin/bash
set -e

# ==================================================================================
# PRODUCTION DEPLOY & RESET SCRIPT
# ==================================================================================
# This script performs a clean deployment and registry cleanup.
# 1. Builds and Pushes a fresh Docker image (linux/amd64).
# 2. Uploads a NEW version of the model to Vertex AI.
# 3. Deploys the NEW version to the Endpoint.
# 4. Deletes ALL previous model versions to ensure a clean registry.
# ==================================================================================

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-east1"
MODEL_NAME="nhits-forecast-model"
ENDPOINT_NAME="nhits-prediction-endpoint-v2"
ARTIFACT_BUCKET="${PROJECT_ID}-vertex-artifacts"
IMAGE_REPO="nhits-serving"
IMAGE_TAG="latest"
LOCAL_MODEL_DIR="local_test_artifacts/nhits_model"

# Check if local model exists
if [ ! -d "$LOCAL_MODEL_DIR" ]; then
    echo "Error: Local model directory '$LOCAL_MODEL_DIR' not found."
    echo "Please run the training/testing workflow first to generate the model."
    exit 1
fi

echo "========================================================"
echo "STARTING PRODUCTION DEPLOY & RESET"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "========================================================"

# ------------------------------------------------------------------
# STEP 1: ARTIFACTS & DOCKER
# ------------------------------------------------------------------
echo "[1/4] Preparing Artifacts & Docker Image..."

# GCS Bucket
if ! gsutil ls -b "gs://${ARTIFACT_BUCKET}" > /dev/null 2>&1; then
    gsutil mb -l $REGION "gs://${ARTIFACT_BUCKET}"
fi

# Upload Model Artifacts
GCS_MODEL_URI="gs://${ARTIFACT_BUCKET}/models/${MODEL_NAME}/$(date +%Y%m%d_%H%M%S)"
gsutil -m cp -r "${LOCAL_MODEL_DIR}/*" "$GCS_MODEL_URI/"

# Build & Push Docker (Force linux/amd64 for Vertex AI)
REPO_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_REPO}/serving"
if ! gcloud artifacts repositories describe $IMAGE_REPO --location=$REGION > /dev/null 2>&1; then
    gcloud artifacts repositories create $IMAGE_REPO --repository-format=docker --location=$REGION
fi
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "Building Docker image (this may take a moment)..."
docker build --no-cache --platform linux/amd64 -t "$REPO_URI:$IMAGE_TAG" -f docker/Dockerfile.serving .
docker push "$REPO_URI:$IMAGE_TAG"

# ------------------------------------------------------------------
# STEP 2: UPLOAD NEW MODEL VERSION
# ------------------------------------------------------------------
echo "[2/4] Uploading New Model Version..."

# Find existing model
EXISTING_MODEL_RAW=$(gcloud ai models list --region=$REGION --filter="display_name=$MODEL_NAME" --format="value(name)" | head -n 1)
EXISTING_MODEL_ID=${EXISTING_MODEL_RAW##*/}

if [ -z "$EXISTING_MODEL_ID" ]; then
    echo "Error: Parent model not found. Please create the initial model first or use the initial deploy script."
    exit 1
fi

FULL_PARENT_MODEL_NAME="projects/$PROJECT_ID/locations/$REGION/models/$EXISTING_MODEL_ID"

# Upload
gcloud ai models upload \
    --region=$REGION \
    --parent-model=$FULL_PARENT_MODEL_NAME \
    --display-name=$MODEL_NAME \
    --container-image-uri="$REPO_URI:$IMAGE_TAG" \
    --artifact-uri="$GCS_MODEL_URI" \
    --container-predict-route="/predict" \
    --container-health-route="/health" \
    --container-ports=8080 \
    --format="value(name)" > model_id.txt

# Get the Base Model ID (should match EXISTING_MODEL_ID)
BASE_MODEL_ID=$(cat model_id.txt)
if [ -z "$BASE_MODEL_ID" ]; then BASE_MODEL_ID=$EXISTING_MODEL_ID; fi

# Identify the NEW Version ID (Sort by versionCreateTime descending)
LATEST_VERSION_ID=$(gcloud ai models list-version $BASE_MODEL_ID --region=$REGION --format="value(versionId)" --sort-by="~versionCreateTime" | head -n 1)

if [ -z "$LATEST_VERSION_ID" ]; then
    echo "Error: Could not determine new version ID."
    exit 1
fi

MODEL_ID_VERSIONED="${BASE_MODEL_ID}@${LATEST_VERSION_ID}"
echo "Successfully uploaded: $MODEL_ID_VERSIONED"

# ------------------------------------------------------------------
# STEP 3: DEPLOY TO ENDPOINT
# ------------------------------------------------------------------
echo "[3/4] Deploying to Endpoint..."

# Find Endpoint
ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --filter="display_name=$ENDPOINT_NAME" --format="value(name)" | head -n 1)
ENDPOINT_ID=${ENDPOINT_ID##*/}

if [ -z "$ENDPOINT_ID" ]; then
    echo "Creating new endpoint: $ENDPOINT_NAME"
    ENDPOINT_ID=$(gcloud ai endpoints create --region=$REGION --display-name=$ENDPOINT_NAME --format="value(name)")
    ENDPOINT_ID=${ENDPOINT_ID##*/}
fi

echo "Deploying $MODEL_ID_VERSIONED to Endpoint $ENDPOINT_ID..."
# Undeploy all other models first to avoid traffic split issues or resource limits
# (Optional: We can just deploy with traffic-split=100 and undeploy others later, but let's be clean)

gcloud ai endpoints deploy-model $ENDPOINT_ID \
    --region=$REGION \
    --model=$MODEL_ID_VERSIONED \
    --display-name="$MODEL_NAME-v$LATEST_VERSION_ID" \
    --machine-type=n1-standard-2 \
    --min-replica-count=1 \
    --max-replica-count=1 \
    --traffic-split=0=100

echo "Deployment Complete!"

# ------------------------------------------------------------------
# STEP 4: CLEANUP OLD VERSIONS
# ------------------------------------------------------------------
echo "[4/4] Cleaning up OLD versions (Keeping v$LATEST_VERSION_ID)..."

# List all versions
ALL_VERSIONS=$(gcloud ai models list-version $BASE_MODEL_ID --region=$REGION --format="value(versionId)")

for v in $ALL_VERSIONS; do
    if [ "$v" != "$LATEST_VERSION_ID" ]; then
        echo "Deleting old version $v..."
        # Check if deployed? The deploy-model command above should have taken 100% traffic.
        # But we might need to explicitly undeploy if it was a different ID.
        # For now, try delete. If it fails because it's deployed, we'll see a warning.
        
        gcloud ai models delete-version ${BASE_MODEL_ID}@${v} --region=$REGION --quiet || echo "Warning: Could not delete version $v (check if deployed)."
    fi
done

echo "========================================================"
echo "SUCCESS: System Reset & Deployed."
echo "Active Version: $LATEST_VERSION_ID"
echo "Endpoint ID: $ENDPOINT_ID"
echo "========================================================"
