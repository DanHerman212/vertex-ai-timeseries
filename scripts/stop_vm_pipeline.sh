#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
else
    echo "Error: .env file not found."
    exit 1
fi

echo "Stopping services on $INSTANCE_NAME..."

gcloud compute ssh $INSTANCE_NAME --project=$PROJECT_ID --zone=$ZONE --command="
    # 1. Stop the Streaming Pipeline (Consumer)
    echo 'Stopping Streaming Pipeline (Python process)...'
    # pkill returns non-zero if no process found, so we add || true
    pkill -f 'streaming/pipeline.py' || echo 'Pipeline was not running.'

    # 2. Stop the Ingestion Service (Producer)
    echo 'Stopping Ingestion Service (Systemd)...'
    sudo systemctl stop mta-ingestion
    # Optional: Disable it so it doesn't start on reboot
    sudo systemctl disable mta-ingestion
    
    echo '-----------------------------------'
    echo 'All pipeline services have been stopped.'
"
