#!/bin/bash
set -e

# Configuration
IMAGE_NAME="nhits-local-test"
CONTAINER_NAME="nhits-test-container"
LOCAL_DATA_DIR="$(pwd)/local_test_data"
mkdir -p "$LOCAL_DATA_DIR"

echo "========================================================"
echo "SETTING UP LOCAL TEST ENVIRONMENT"
echo "========================================================"

# 1. Create Dummy Data
echo "Generating dummy data..."
cat <<EOF > "$LOCAL_DATA_DIR/raw_data.csv"
arrival_date,duration,mbt,dow
2024-01-01 00:00:00,10,5,0
2024-01-01 01:00:00,12,6,0
2024-01-01 02:00:00,11,5,0
2024-01-01 03:00:00,13,7,0
2024-01-01 04:00:00,10,5,0
EOF

# Generate 1000 rows of dummy data
for i in {1..1000}; do
    # Increment hour
    DATE=$(date -v+${i}H -j -f "%Y-%m-%d %H:%M:%S" "2024-01-01 00:00:00" "+%Y-%m-%d %H:%M:%S")
    echo "$DATE,10,5,0" >> "$LOCAL_DATA_DIR/raw_data.csv"
done

# 2. Build Docker Image (Native Architecture)
echo "Building Docker image ($IMAGE_NAME)..."
docker build -t $IMAGE_NAME -f docker/Dockerfile.local_test .

# 3. Run Pipeline Steps inside Container
echo "Running Pipeline Steps..."

# We run the container interactively to execute multiple commands
docker run --rm \
    -v "$LOCAL_DATA_DIR":/data \
    $IMAGE_NAME \
    bash -c "
    set -e
    
    echo '--- STEP 1: PREPROCESS ---'
    python src/preprocess.py \
        --input_csv /data/raw_data.csv \
        --output_csv /data/processed.csv
        
    echo '--- STEP 2: TRAIN (Short) ---'
    # This will train a real model on small data
    python src/train_nhits.py \
        --input_csv /data/processed.csv \
        --model_dir /data/model \
        --test_output_csv /data/test.csv
        
    echo '--- STEP 3: EVALUATE ---'
    # This is the step that was failing
    python src/evaluate_nhits.py \
        --test_dataset_path /data/test.csv \
        --model_dir /data/model \
        --metrics_output_path /data/metrics.json \
        --plot_output_path /data/loss.html \
        --prediction_plot_path /data/pred.html
        
    echo '--- SUCCESS: PIPELINE FINISHED ---'
"

echo "========================================================"
echo "TEST COMPLETED SUCCESSFULLY"
echo "Artifacts are in: $LOCAL_DATA_DIR"
echo "========================================================"
