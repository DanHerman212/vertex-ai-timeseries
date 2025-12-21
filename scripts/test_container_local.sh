#!/bin/bash
set -e

# Define paths
IMAGE_NAME="nhits-local-test"
INPUT_DATA_HOST="$(pwd)/local_test_data"
ARTIFACTS_DIR_HOST="$(pwd)/local_test_artifacts"

# Container paths
INPUT_DATA_CONTAINER="/data/mini_data.csv"
ARTIFACTS_DIR_CONTAINER="/artifacts"

# Clean up previous artifacts
rm -rf local_test_artifacts
mkdir -p local_test_artifacts

echo "========================================================"
echo "STEP 0: Building Docker Image (Local Test Variant)"
echo "========================================================"
# Use the local test Dockerfile which uses python:3.10-slim (works on ARM64)
# instead of the Vertex AI GPU image (which crashes on ARM64)
docker build -t $IMAGE_NAME -f docker/Dockerfile.local_test .

echo "========================================================"
echo "STEP 1: Training NHITS Model (Containerized)"
echo "========================================================"

echo "DEBUG: Checking container environment..."
docker run --rm $IMAGE_NAME python --version

echo "DEBUG: Starting Training..."
docker run --rm \
    -v "$INPUT_DATA_HOST":/data \
    -v "$ARTIFACTS_DIR_HOST":/artifacts \
    $IMAGE_NAME \
    python src/train_nhits.py \
    --input_csv "$INPUT_DATA_CONTAINER" \
    --model_dir "$ARTIFACTS_DIR_CONTAINER/nhits_model" \
    --df_output_csv "$ARTIFACTS_DIR_CONTAINER/full_df.csv" \
    --logs_dir "$ARTIFACTS_DIR_CONTAINER/logs"

echo "========================================================"
echo "STEP 2: Evaluating NHITS Model (Containerized)"
echo "========================================================"
docker run --rm \
    -v "$ARTIFACTS_DIR_HOST":/artifacts \
    $IMAGE_NAME \
    python src/evaluate_nhits.py \
    --df_csv_path "$ARTIFACTS_DIR_CONTAINER/full_df.csv" \
    --model_dir "$ARTIFACTS_DIR_CONTAINER/nhits_model" \
    --metrics_output_path "$ARTIFACTS_DIR_CONTAINER/metrics.json" \
    --html_output_path "$ARTIFACTS_DIR_CONTAINER/report.html" \
    --logs_dir "$ARTIFACTS_DIR_CONTAINER/logs"

echo "========================================================"
echo "Container test completed successfully!"
echo "Report generated at: local_test_artifacts/report.html"
echo "========================================================"
