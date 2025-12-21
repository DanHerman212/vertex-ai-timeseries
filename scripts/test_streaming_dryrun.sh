#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
else
    echo "Error: .env file not found."
    exit 1
fi

# Required Variables
REQUIRED_VARS=("PROJECT_ID" "ZONE" "INSTANCE_NAME" "SUBSCRIPTION_ID")
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo "Error: $VAR is not set in .env"
        exit 1
    fi
done

echo "========================================================"
echo "TEST: Running Streaming Pipeline in DRY RUN Mode"
echo "Target: $INSTANCE_NAME ($ZONE)"
echo "Subscription: $SUBSCRIPTION_ID"
echo "========================================================"

# We use a dummy endpoint ID because dry_run skips the actual call
DUMMY_ENDPOINT="1234567890"

gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE --command="
    cd nhits_workflow
    source venv/bin/activate
    
    echo 'Starting Streaming Pipeline (Dry Run)...'
    echo 'Press Ctrl+C to stop.'
    
    # Run as a module to ensure imports work correctly
    python3 -m streaming.pipeline \
        --project_id=$PROJECT_ID \
        --region=us-east1 \
        --input_subscription=$SUBSCRIPTION_ID \
        --endpoint_id=$DUMMY_ENDPOINT \
        --weather_csv=weather_data.csv \
        --dry_run
"
