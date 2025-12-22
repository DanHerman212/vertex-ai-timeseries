#!/bin/bash
set -e

IMAGE_NAME="nhits-serving-local"
CONTAINER_NAME="nhits-serving-test"
PORT=8080

echo "========================================================"
echo "TEST: Building Serving Image Locally"
echo "========================================================"

# Build the image using the serving Dockerfile
# We use the same Dockerfile as production to ensure fidelity
docker build -t $IMAGE_NAME -f docker/Dockerfile.serving .

echo "========================================================"
echo "TEST: Starting Container"
echo "========================================================"

# Stop any existing container
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Run the container
# We mount the local model artifacts to /app/nhits_model because serve.py defaults to looking there
# if AIP_STORAGE_URI is not set.
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8080 \
    -v "$(pwd)/local_test_artifacts/nhits_model:/app/nhits_model" \
    $IMAGE_NAME

echo "Waiting for container to start (10s)..."
sleep 10

# Check if container is still running
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "Error: Container died immediately. Logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

echo "========================================================"
echo "TEST: Health Check"
echo "========================================================"
curl -v http://localhost:$PORT/health

echo ""
echo "========================================================"
echo "TEST: Prediction"
echo "========================================================"

# Create a dummy payload with required features
# We use python to generate the JSON because it's cleaner
python3 -c '
import json
import pandas as pd
import numpy as np

# Generate timestamps
dates = pd.date_range(start="2025-01-01 10:00:00", periods=161, freq="15min")

data = []
for i, date in enumerate(dates):
    row = {
        "ds": str(date),
        "unique_id": "L",
        "y": 10.0 + np.sin(i/10.0), # Dummy target
        # Required Exogenous Features
        "temp": 70.0,
        "precip": 0.0,
        "snow": 0.0,
        "snowdepth": 0.0,
        "visibility": 10.0,
        "windspeed": 5.0,
        "dow": 1 if date.weekday() >= 5 else 0,
        # Historical Features (needed for model input size)
        "rolling_mean_10": 10.0,
        "rolling_std_10": 1.0,
        "rolling_mean_50": 10.0,
        "rolling_std_50": 1.0,
        "rolling_max_10": 12.0,
        "duration": 15.0
    }
    data.append(row)

payload = {"instances": data}
print(json.dumps(payload))
' > payload.json

curl -X POST \
    -H "Content-Type: application/json" \
    -d @payload.json \
    http://localhost:$PORT/predict

echo ""
echo "========================================================"
echo "Logs:"
docker logs $CONTAINER_NAME | head -n 20

echo "========================================================"
echo "Cleaning up..."
docker rm -f $CONTAINER_NAME
rm payload.json
