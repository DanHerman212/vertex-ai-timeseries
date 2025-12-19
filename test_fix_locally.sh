#!/bin/bash
set -e

# Setup directories
mkdir -p local_test_data
mkdir -p local_test_model

# 1. Preprocess
echo "Running Preprocessing..."
python src/preprocess.py \
    --input_csv training_and_preprocessing_workflows/ml_dec_17.csv \
    --output_csv local_test_data/processed.csv

# 2. Train (Fast, with max_steps=5)
echo "Running Training..."
python src/train_nhits.py \
    --input_csv local_test_data/processed.csv \
    --model_dir local_test_model \
    --test_output_csv local_test_data/test.csv

# 3. Evaluate (The part we want to test)
echo "Running Evaluation..."
python src/evaluate_nhits.py \
    --test_dataset_path local_test_data/test.csv \
    --model_dir local_test_model \
    --metrics_output_path local_test_data/metrics.json \
    --plot_output_path local_test_data/loss_plot.html \
    --prediction_plot_path local_test_data/pred_plot.html

echo "Test Complete!"
