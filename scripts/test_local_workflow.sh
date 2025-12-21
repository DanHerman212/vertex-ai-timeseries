#!/bin/bash
set -e

# Define paths
INPUT_DATA="local_test_data/mini_data.csv"
MODEL_DIR="local_test_artifacts/nhits_model"
FULL_DF_OUTPUT="local_test_artifacts/full_df.csv"
LOGS_DIR="local_test_artifacts/logs"
METRICS_OUTPUT="local_test_artifacts/metrics.json"
HTML_OUTPUT="local_test_artifacts/report.html"

# Clean up previous run
rm -rf local_test_artifacts
mkdir -p local_test_artifacts

echo "========================================================"
echo "STEP 1: Training NHITS Model (Local Test)"
echo "========================================================"
python src/train_nhits.py \
    --input_csv "$INPUT_DATA" \
    --model_dir "$MODEL_DIR" \
    --df_output_csv "$FULL_DF_OUTPUT" \
    --logs_dir "$LOGS_DIR"

echo ""
echo "========================================================"
echo "STEP 2: Evaluating NHITS Model (Local Test)"
echo "========================================================"
python src/evaluate_nhits.py \
    --model_dir "$MODEL_DIR" \
    --df_csv_path "$FULL_DF_OUTPUT" \
    --metrics_output_path "$METRICS_OUTPUT" \
    --html_output_path "$HTML_OUTPUT" \
    --logs_dir "$LOGS_DIR"

echo ""
echo "========================================================"
echo "Local test completed successfully!"
echo "Report generated at: $HTML_OUTPUT"
echo "========================================================"
