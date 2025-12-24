#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Required Variables
REQUIRED_VARS=("PROJECT_ID" "ENDPOINT_ID" "SUBSCRIPTION_ID" "BUCKET_NAME" "REGION")
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo "Error: $VAR is not set. Please set it in .env or export it."
        exit 1
    fi
done

echo "🚀 Deploying Streaming Pipeline to Dataflow (Production)..."

# Generate a unique job name
JOB_NAME="nhits-streaming-$(date +%Y%m%d-%H%M%S)"

# Ensure setup.py exists
if [ ! -f setup.py ]; then
    echo "Error: setup.py not found. Please run this script from the root of the repository."
    exit 1
fi

# Run as a module to ensure imports work correctly
# Added --prebuild_sdk_container_engine=cloud_build to speed up worker startup
python3 -m streaming.pipeline \
    --runner=DataflowRunner \
    --project=$PROJECT_ID \
    --region=$REGION \
    --temp_location=gs://$BUCKET_NAME/temp \
    --staging_location=gs://$BUCKET_NAME/staging \
    --job_name=$JOB_NAME \
    --setup_file=./setup.py \
    --prebuild_sdk_container_engine=cloud_build \
    --input_subscription=projects/$PROJECT_ID/subscriptions/$SUBSCRIPTION_ID \
    --endpoint_id=$ENDPOINT_ID \
    --project_id=$PROJECT_ID \
    --weather_api_key=$WEATHER_API_KEY

echo "✅ Job submitted: $JOB_NAME"
echo "Monitor at: https://console.cloud.google.com/dataflow/jobs?project=$PROJECT_ID"
