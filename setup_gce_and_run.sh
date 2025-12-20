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

# 0. Ensure Pub/Sub Resources Exist
echo "Checking Pub/Sub resources..."
if ! gcloud pubsub topics describe $TOPIC_ID --project=$PROJECT_ID > /dev/null 2>&1; then
    echo "Creating topic $TOPIC_ID..."
    gcloud pubsub topics create $TOPIC_ID --project=$PROJECT_ID
else
    echo "Topic $TOPIC_ID already exists."
fi

if ! gcloud pubsub subscriptions describe $SUBSCRIPTION_ID --project=$PROJECT_ID > /dev/null 2>&1; then
    echo "Creating subscription $SUBSCRIPTION_ID..."
    gcloud pubsub subscriptions create $SUBSCRIPTION_ID --topic=$TOPIC_ID --project=$PROJECT_ID
else
    echo "Subscription $SUBSCRIPTION_ID already exists."
fi

# 1. Create/Start GCE Instance
echo "Configuring GCE instance..."
if gcloud compute instances describe $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE > /dev/null 2>&1; then
    echo "Instance $INSTANCE_NAME exists. Ensuring it is running..."
    gcloud compute instances start $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE || true
else
    echo "Creating instance $INSTANCE_NAME..."
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
    
    # --- Setup Ingestion Service (Producer) ---
    echo "Setting up Ingestion Service..."
    sudo mkdir -p /opt/mta-ingestion
    
    # Copy files to /opt/mta-ingestion
    sudo cp ingestion/ingest_feed.py /opt/mta-ingestion/
    sudo cp ingestion/requirements.txt /opt/mta-ingestion/
    sudo cp .env /opt/mta-ingestion/
    
    # Setup Venv for Ingestion
    cd /opt/mta-ingestion
    sudo python3 -m venv venv
    sudo ./venv/bin/pip install -r requirements.txt
    
    # Setup Systemd Service
    cd ~/nhits_workflow
    # Update User in service file to root (simple fix for GCE)
    sed -i 's/User=google-sudoers/User=root/g' ingestion/mta-ingestion.service
    
    sudo cp ingestion/mta-ingestion.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable mta-ingestion
    sudo systemctl restart mta-ingestion
    
    echo "Ingestion Service (Producer) started in background."
    
    # --- Run Streaming Pipeline (Consumer) ---
    echo 'Starting Streaming Pipeline (Consumer)...'
    echo 'Press Ctrl+C to stop after observing 50-100 trains.'
    
    # Go back to workspace
    cd ~/nhits_workflow
    
    python3 streaming/pipeline.py \
        --project_id=$PROJECT_ID \
        --region=us-east1 \
        --endpoint_id=$ENDPOINT_ID \
        --input_subscription=projects/$PROJECT_ID/subscriptions/$SUBSCRIPTION_ID \
        --weather_api_key=$WEATHER_API_KEY \
        --dry_run
"
