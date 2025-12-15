import os
import argparse
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE, MQLoss
from google.cloud import storage

# 1. Load Data
def load_data(input_path=None):
    print("Loading data...")
    if input_path and os.path.exists(input_path):
        print(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        return df
    
    print("Generating synthetic data...")
    # ... (keep synthetic generation if needed, but updated to match new schema)
    # For brevity, assuming input_path is provided in this workflow
    raise ValueError("Input path must be provided")

def train_and_save(output_path, input_path=None):
    Y_df = load_data(input_path)
    
    # Add unique_id required by NeuralForecast
    if 'unique_id' not in Y_df.columns:
        Y_df['unique_id'] = 'E'
    
    # Split logic (60/20/20)
    n = len(Y_df)
    train_size = int(n * 0.6)
    val_size = int(n * 0.2)
    # test_size = n - train_size - val_size # Implicit in remaining data
    
    # Define Exogenous Variables
    futr_exog_list = ['week_sin', 'week_cos', 'day_sin', 'day_cos'] # Add weather if available in input
    hist_exog_list = ['rolling_mean_10', 'rolling_std_10', 'rolling_mean_50', 'rolling_std_50', 'rolling_max_10']
    
    # 2. Define Model
    # NHITS Configuration from notebook
    models = [
        NHITS(
            h=1,                      # Horizon: Predict next step
            input_size=160,           # Lookback window
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), 
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            max_steps=1000,           
            early_stop_patience_steps=10,
            scaler_type='robust',     
            learning_rate=1e-3,
            n_pool_kernel_size=[2, 2, 2], 
            n_freq_downsample=[168, 24, 1] 
        )
    ]
    
    # Initialize NeuralForecast
    # Assuming 'H' freq, or let it infer/use integer index if irregular
    nf = NeuralForecast(models=models, freq='H')
    
    # 3. Train Model
    print("Training NHITS model...")
    # Use val_size for early stopping
    nf.fit(df=Y_df, val_size=val_size)
    
    # 4. Save Model
    print(f"Saving model to {output_path}...")
    nf.save(path=output_path, model_index=None, overwrite=True)

def upload_to_gcs(local_path, bucket_name, blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Upload the directory
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            remote_path = os.path.join(blob_path, os.path.relpath(local_file, local_path))
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
            print(f"Uploaded {local_file} to gs://{bucket_name}/{remote_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, help='Path to input CSV file')
    parser.add_argument('--bucket_name', type=str, help='GCS Bucket name')
    args = parser.parse_args()

    # Environment variables usually set by Vertex AI or user
    BUCKET_NAME = args.bucket_name or os.getenv("GCS_BUCKET_NAME", "my-bucket")
    MODEL_DIR = "nhits_model"
    
    train_and_save(MODEL_DIR, args.input_csv)
    
    # If running in cloud, upload artifacts
    if os.getenv("CLOUD_RUN") or args.bucket_name:
        upload_to_gcs(MODEL_DIR, BUCKET_NAME, "models/nhits/v1")
