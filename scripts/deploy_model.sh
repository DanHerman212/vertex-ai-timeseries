#!/bin/bash
set -e

# ==================================================================================
# DEPLOY MODEL TO VERTEX AI ENDPOINT
# ==================================================================================
# This script:
# 1. Uploads the local model artifacts to GCS.
# 2. Builds and pushes the serving container image.
# 3. Uploads the model to Vertex AI Model Registry.
# 4. Creates a Vertex AI Endpoint.
# 5. Deploys the model to the Endpoint.

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-east1"  # Default region
MODEL_NAME="nhits-forecast-model"
ENDPOINT_NAME="nhits-prediction-endpoint"
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
echo "Deploying Model to Vertex AI"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Model Dir: $LOCAL_MODEL_DIR"
echo "========================================================"

# 1. Create Bucket (if not exists)
echo "[1/5] Checking/Creating GCS Bucket..."
if ! gsutil ls -b "gs://${ARTIFACT_BUCKET}" > /dev/null 2>&1; then
    gsutil mb -l $REGION "gs://${ARTIFACT_BUCKET}"
    echo "Created bucket gs://${ARTIFACT_BUCKET}"
else
    echo "Bucket gs://${ARTIFACT_BUCKET} already exists."
fi

# 2. Upload Model Artifacts
GCS_MODEL_URI="gs://${ARTIFACT_BUCKET}/models/${MODEL_NAME}/$(date +%Y%m%d_%H%M%S)"
echo "[2/5] Uploading model artifacts to $GCS_MODEL_URI..."
gsutil -m cp -r "${LOCAL_MODEL_DIR}/*" "$GCS_MODEL_URI/"

# 3. Build and Push Serving Image
REPO_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_REPO}/serving"
echo "[3/5] Building and Pushing Docker Image to $REPO_URI..."

# Create repository if it doesn't exist
if ! gcloud artifacts repositories describe $IMAGE_REPO --location=$REGION > /dev/null 2>&1; then
    gcloud artifacts repositories create $IMAGE_REPO --repository-format=docker --location=$REGION
fi

# Configure docker auth
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build and Push
docker build -t "$REPO_URI:$IMAGE_TAG" -f docker/Dockerfile.serving .
docker push "$REPO_URI:$IMAGE_TAG"

# 4. Upload Model to Vertex AI
echo "[4/5] Uploading Model to Vertex AI Registry..."

# Check for existing model to upload as a new version
EXISTING_MODEL_NAME=$(gcloud ai models list --region=$REGION --filter="display_name=$MODEL_NAME" --format="value(name)" | head -n 1)

if [ -n "$EXISTING_MODEL_NAME" ]; then
    echo "Found existing model: $EXISTING_MODEL_NAME"
    echo "Uploading as a new version..."
    
    gcloud ai models upload \
        --region=$REGION \
        --parent-model=$EXISTING_MODEL_NAME \
        --display-name=$MODEL_NAME \
        --container-image-uri="$REPO_URI:$IMAGE_TAG" \
        --artifact-uri="$GCS_MODEL_URI" \
        --container-predict-route="/predict" \
        --container-health-route="/health" \
        --container-ports=8080 \
        --format="value(name)" > model_id.txt
        
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Error: Model version upload failed."
        rm model_id.txt
        exit 1
    fi
    
    # When uploading a version, the output name is the Model ID (not versioned).
    # We need to fetch the latest version ID.
    BASE_MODEL_ID=$(cat model_id.txt)

    # Fallback: If output is empty, use the known existing model ID
    if [ -z "$BASE_MODEL_ID" ]; then
        echo "Warning: Command output empty. Using known Model ID: $EXISTING_MODEL_NAME"
        BASE_MODEL_ID=$EXISTING_MODEL_NAME
    fi
    
    # Fetch the latest version ID (sort by createTime descending)
    LATEST_VERSION_ID=$(gcloud ai models list-versions $BASE_MODEL_ID --region=$REGION --format="value(versionId)" --sort-by="~createTime" | head -n 1)
    
    if [ -z "$LATEST_VERSION_ID" ]; then
        echo "Error: Could not determine new version ID."
        exit 1
    fi
    
    # Construct the versioned model ID for deployment
    MODEL_ID="${BASE_MODEL_ID}@${LATEST_VERSION_ID}"
    echo "Uploaded new version: $MODEL_ID"

else
    echo "Creating new model..."
    
    gcloud ai models upload \
        --region=$REGION \
        --display-name=$MODEL_NAME \
        --container-image-uri="$REPO_URI:$IMAGE_TAG" \
        --artifact-uri="$GCS_MODEL_URI" \
        --container-predict-route="/predict" \
        --container-health-route="/health" \
        --container-ports=8080 \
        --format="value(name)" > model_id.txt

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Error: Model upload failed."
        rm model_id.txt
        exit 1
    fi

    MODEL_ID=$(cat model_id.txt)
    
    # Fallback: If output is empty, try to find the model by name
    if [ -z "$MODEL_ID" ]; then
        echo "Warning: Command output empty. Searching for model by name..."
        MODEL_ID=$(gcloud ai models list --region=$REGION --filter="display_name=$MODEL_NAME" --format="value(name)" --sort-by="~createTime" | head -n 1)
    fi

    echo "Model uploaded. ID: $MODEL_ID"
fi

if [ -z "$MODEL_ID" ]; then
    echo "Error: Failed to capture MODEL_ID. Model upload may have failed (even with exit code 0)."
    rm model_id.txt
    exit 1
fi

rm model_id.txt

# 5. Create Endpoint and Deploy
echo "[5/5] Creating Endpoint and Deploying Model..."

# Check if endpoint exists
EXISTING_ENDPOINT_ID=$(gcloud ai endpoints list \
    --region=$REGION \
    --filter="display_name=$ENDPOINT_NAME" \
    --format="value(name)" | head -n 1)

if [ -z "$EXISTING_ENDPOINT_ID" ]; then
    echo "Creating new endpoint..."
    ENDPOINT_ID=$(gcloud ai endpoints create \
        --region=$REGION \
        --display-name=$ENDPOINT_NAME \
        --format="value(name)")
    
    if [ -z "$ENDPOINT_ID" ]; then
        echo "Error: Failed to capture ENDPOINT_ID."
        exit 1
    fi
else
    echo "Using existing endpoint: $EXISTING_ENDPOINT_ID"
    ENDPOINT_ID=$EXISTING_ENDPOINT_ID
fi

echo "Deploying model to endpoint $ENDPOINT_ID..."
gcloud ai endpoints deploy-model "$ENDPOINT_ID" \
    --region=$REGION \
    --model="$MODEL_ID" \
    --display-name="$MODEL_NAME-deployment" \
    --machine-type="n1-standard-2" \
    --min-replica-count=1 \
    --max-replica-count=1 \
    --traffic-split="0=100"

echo "========================================================"
echo "Deployment Complete!"
echo "Endpoint ID: $ENDPOINT_ID"
echo "You can now update your pipeline configuration to use this endpoint."
echo "========================================================"
