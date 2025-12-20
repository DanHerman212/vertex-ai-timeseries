#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
else
    echo "Error: .env file not found. Please copy .env.example to .env and fill in the values."
    exit 1
fi

# Required Variables
REQUIRED_VARS=("PROJECT_ID" "ZONE" "INSTANCE_NAME" "WEATHER_API_KEY" "ENDPOINT_ID" "TOPIC_ID" "SUBSCRIPTION_ID")
for VAR in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo "Error: $VAR is not set in .env"
        exit 1
    fi
done

# 1. Create GCE Instance
echo "Creating GCE instance..."
# Check if instance exists
if gcloud compute instances describe $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE > /dev/null 2>&1; then
    echo "Instance $INSTANCE_NAME already exists. Skipping creation."
else
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=e2-standard-2 \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --image-family=debian-11 \
        --image-project=debian-cloud
fi

echo "Waiting for instance to be ready..."
sleep 30

# 2. Copy files
echo "Copying files to instance..."
tar -czf workspace.tar.gz --exclude='venv' --exclude='.git' --exclude='__pycache__' .
gcloud compute scp workspace.tar.gz $INSTANCE_NAME:~/ --project=$PROJECT_ID --zone=$ZONE
rm workspace.tar.gz

# 3. Setup and Run
echo "Setting up and running on instance..."
gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE --command="
    sudo apt-get update
    sudo apt-get install -y python3-pip python3-venv
    
    mkdir -p nhits_workflow
    tar -xzf workspace.tar.gz -C nhits_workflow
    cd nhits_workflow
    
    python3 -m venv venv
    source venv/bin/activate
    
    pip install -r requirements.txt
    
    # Create Subscription if not exists
    # We assume the topic 'vehicle-position-updates' exists. If not, this will fail.
    # If the topic doesn't exist, we can't test the live feed anyway unless westart ingestion.

    gcloud pubsub subscriptions create $SUBSCRIPTION_ID --topic=$TOPIC_ID || true
    
    echo 'Starting Streaming Pipeline...'
    echo 'Press Ctrl+C to stop after observing 50-100 trains.'
    
    python3 streaming/pipeline.py \
        --project_id=$PROJECT_ID \
        --region=us-east1 \
        --endpoint_id=$ENDPOINT_ID \
        --input_subscription=projects/$PROJECT_ID/subscriptions/$SUBSCRIPTION_ID \
        --weather_api_key=$WEATHER_API_KEY \
        --dry_run
"
