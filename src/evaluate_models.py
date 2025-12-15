import argparse
import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf
from neuralforecast import NeuralForecast
from google.cloud import storage
from sklearn.metrics import mean_absolute_error

# Import GRU helpers (assuming train_gru.py is in the same directory)
# We need to make sure train_gru.py is importable. 
# If not, we can duplicate the simple loading logic to avoid dependency issues in some environments.
def load_data_manual(input_path):
    with open(input_path) as f:
        data = f.read()
    lines = [line for line in data.split("\n") if line.strip()]
    header = lines[0].split(",")
    lines = lines[1:]
    
    mbt = np.zeros((len(lines),))
    raw_data = np.zeros((len(lines), len(header) - 1))
    
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(",")[1:]]
        mbt[i] = values[1] 
        raw_data[i, :] = values[:]
        
    return mbt, raw_data

def evaluate_gru(model_path, scaler_path, raw_data, mbt, test_start_idx, sequence_length=150):
    print("Evaluating GRU Model...")
    # Load Model
    model = tf.keras.models.load_model(model_path)
    
    # Load Scaler
    scaler = joblib.load(scaler_path)
    train_mean = scaler['mean']
    train_std = scaler['std']
    
    # Scale Data
    raw_data_scaled = (raw_data - train_mean) / train_std
    
    # Prepare Test Data (Sliding Window)
    # We need to start 'sequence_length' steps before the test_start_idx to predict the first test point
    start_idx = test_start_idx - sequence_length
    
    if start_idx < 0:
        raise ValueError("Not enough data history for the first test point.")

    # Create dataset
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=raw_data_scaled[start_idx:-1], # Input up to the last point
        targets=mbt[test_start_idx:],       # Targets starting from test_start_idx
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=128,
        shuffle=False
    )
    
    # Predict
    predictions_scaled = model.predict(test_ds, verbose=0)
    
    # Inverse Transform (Target is mbt, which is index 1 in raw_data, but index 1 in scaler mean/std too)
    # Note: The model output is shape (N, 1). We need to unscale it using the specific mean/std of the target column.
    # mbt was at index 1 of the features passed to scale_data in train_gru.py
    target_col_idx = 1 
    pred_mean = train_mean[target_col_idx]
    pred_std = train_std[target_col_idx]
    
    predictions = (predictions_scaled * pred_std) + pred_mean
    
    # Get Actuals
    # The dataset targets are already aligned
    actuals = np.concatenate([y for x, y in test_ds], axis=0)
    
    mae = mean_absolute_error(actuals, predictions)
    print(f"GRU MAE: {mae}")
    return mae

def evaluate_nhits(model_path, df, test_start_idx):
    print("Evaluating NHITS Model...")
    # Load Model
    nhits = NeuralForecast.load(path=model_path)
    
    # Prepare Test Data
    # NeuralForecast expects a DataFrame with 'ds', 'y', 'unique_id'
    # We will use the 'predict' method. 
    # For a fair comparison on the test set, we should ideally perform a rolling forecast.
    # However, for this implementation, we will use the model to predict the test period.
    
    # Filter for test period
    test_df = df.iloc[test_start_idx:].copy()
    
    # To predict for the test period, we need the history leading up to it.
    # NeuralForecast's predict() uses the data stored in the object or passed to it.
    # If we loaded the model, it might not have the data.
    # We need to pass the full dataset to predict, but mask the future?
    # Actually, the easiest way to get predictions for the test set with a pre-trained model 
    # is to pass the historical data up to the split point and ask for a forecast of length len(test_set).
    
    history_df = df.iloc[:test_start_idx].copy()
    horizon = len(test_df)
    
    # Predict
    # Note: This predicts 'horizon' steps into the future from the end of 'history_df'
    fcst_df = nhits.predict(df=history_df, h=horizon)
    
    # Align with Actuals
    # fcst_df will have 'ds' and 'NHITS'. We merge with test_df on 'ds'
    merged = pd.merge(test_df, fcst_df, on=['ds', 'unique_id'], how='inner')
    
    mae = mean_absolute_error(merged['y'], merged['NHITS'])
    print(f"NHITS MAE: {mae}")
    return mae

def promote_model(winner_name, metrics, bucket_name=None):
    print(f"\n*** WINNER: {winner_name} ***")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    deployment_info = {
        "active_model": winner_name,
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Save locally
    with open("deployment_config.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
        
    # Upload to GCS if bucket provided
    if bucket_name:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("config/deployment_config.json")
        blob.upload_from_filename("deployment_config.json")
        print(f"Updated deployment config in gs://{bucket_name}/config/deployment_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gru_model_dir', type=str, required=True)
    parser.add_argument('--nhits_model_dir', type=str, required=True)
    parser.add_argument('--test_data_csv', type=str, required=True)
    parser.add_argument('--bucket_name', type=str, help='GCS Bucket for promotion')
    args = parser.parse_args()

    # 1. Load Data (Common Source)
    # We load as Pandas for NHITS and Manual for GRU to ensure consistency with training
    df = pd.read_csv(args.test_data_csv)
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
    if 'unique_id' not in df.columns:
        df['unique_id'] = '1' # Default ID if not present
        
    mbt, raw_data = load_data_manual(args.test_data_csv)
    
    # 2. Define Split (Last 20%)
    n = len(df)
    test_start_idx = int(n * 0.8)
    
    metrics = {}
    
    # 3. Evaluate GRU
    try:
        gru_path = os.path.join(args.gru_model_dir, "gru_model.keras")
        scaler_path = os.path.join(args.gru_model_dir, "scaler.pkl")
        metrics['GRU'] = evaluate_gru(gru_path, scaler_path, raw_data, mbt, test_start_idx)
    except Exception as e:
        print(f"Failed to evaluate GRU: {e}")
        metrics['GRU'] = float('inf')

    # 4. Evaluate NHITS
    try:
        metrics['NHITS'] = evaluate_nhits(args.nhits_model_dir, df, test_start_idx)
    except Exception as e:
        print(f"Failed to evaluate NHITS: {e}")
        metrics['NHITS'] = float('inf')
        
    # 5. Compare and Promote
    if metrics['GRU'] < metrics['NHITS']:
        winner = "GRU"
    else:
        winner = "NHITS"
        
    promote_model(winner, metrics, args.bucket_name)
