import os
import argparse
import pandas as pd
import numpy as np
import torch
import json
import base64
from io import BytesIO
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE, MQLoss
from neuralforecast.losses.numpy import mae, rmse
from pytorch_lightning.loggers import CSVLogger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Load and Prepare Data
def load_data(input_path):
    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found.")
        
    df = pd.read_csv(input_path)
    
    # Rename columns to NeuralForecast convention
    # arrival_date -> ds, mbt -> y
    if 'arrival_date' in df.columns:
        df = df.rename(columns={'arrival_date': 'ds'})
    if 'mbt' in df.columns:
        df = df.rename(columns={'mbt': 'y'})
        
    # Ensure datetime
    df['ds'] = pd.to_datetime(df['ds'])
    # Remove timezone if present
    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_localize(None)
        
    # Do NOT round to hourly frequency. Use raw timestamps.
    # df['ds'] = df['ds'].dt.floor('H')
    
    # Add unique_id
    df['unique_id'] = 'E'
    
    # Ensure data is sorted by time
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    
    print(f"Data shape after processing: {df.shape}")
    return df

def train_and_save(model_dir, input_path, test_output_path=None, max_steps=1000):
    # 1. Load Data
    Y_df = load_data(input_path)
    
    # Split logic (60/20/20)
    n = len(Y_df)
    train_size = int(n * 0.6)
    val_size = int(n * 0.2)
    test_size = n - train_size - val_size

    train_df = Y_df.iloc[:train_size]
    val_df = Y_df.iloc[train_size:train_size+val_size]
    test_df = Y_df.iloc[train_size+val_size:]

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Create Train+Val DataFrame for fitting
    train_val_df = pd.concat([train_df, val_df])
    
    # Create Test DataFrame for export
    # IMPORTANT: We include the lookback window (input_size=160) so the model has context
    # for the first prediction in the test set.
    # We also add a buffer (e.g. 50 steps) to allow for validation splits during evaluation
    # (since evaluate_nhits.py uses val_size=0, we need at least input_size + horizon history)
    input_size = 160
    buffer_size = 50
    start_idx = max(0, train_size + val_size - input_size - buffer_size)
    test_df_export = Y_df.iloc[start_idx:]
    
    if test_output_path:
        print(f"Saving test dataframe to {test_output_path}...")
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
        test_df_export.to_csv(test_output_path, index=False)
    
    # Define Exogenous Variables
    # Note: futr_exog_list requires these columns to be known in the future. 
    # Since they are calendar based, they are fine.
    # Removed cyclic features as per user request
    # Added 'dow' (0=Weekday, 1=Weekend)
    futr_exog_list = ['temp', 'precip', 'snow', 'snowdepth', 'visibility', 'windspeed', 'dow']
    # Added 'duration' to hist_exog_list as it is a feature, not target
    hist_exog_list = ['rolling_mean_10', 'rolling_std_10', 'rolling_mean_50', 'rolling_std_50', 'rolling_max_10', 'duration']
    
    # Setup Logger
    # Use a temporary directory for logs to prevent NeuralForecast.save(overwrite=True) from deleting them
    import tempfile
    import shutil
    temp_log_dir = tempfile.mkdtemp()
    logger = CSVLogger(save_dir=temp_log_dir, name="training_logs")

    # 2. Define Model
    models = [
        NHITS(
            h=1,                      # Horizon: Predict next step
            input_size=160,           # Lookback window (Aligned with Notebook)
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), 
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            max_steps=max_steps,           
            early_stop_patience_steps=10,
            val_check_steps=100,      # Check validation every 100 steps
            batch_size=256,           # Batch size (Aligned with GRU)
            scaler_type='standard',   # Changed to standard to match GRU model
            learning_rate=1e-3,
            n_pool_kernel_size=[2, 2, 2], 
            n_freq_downsample=[168, 24, 1],
            accelerator="gpu" if torch.cuda.is_available() else "cpu"
            # logger=logger      # Removed to prevent path issues during loading
        )
    ]
    
    # Initialize NeuralForecast
    # Use freq='S' (Second) to accommodate high-frequency/irregular data without aggregation
    nf = NeuralForecast(models=models, freq='S')
    
    # 3. Train Model
    print("Training NHITS model...")
    nf.fit(df=train_val_df, val_size=val_size)
    
    # 4. Save Model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print(f"Saving model to {model_dir}...")
    nf.save(path=model_dir, model_index=None, overwrite=True)
    
    # Cleanup temp logs if they were created (but we removed logger injection)
    if os.path.exists(temp_log_dir):
        shutil.rmtree(temp_log_dir)
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model_dir', type=str, default='nhits_model', help='Local directory to save model')
    parser.add_argument('--test_output_csv', type=str, required=False, help='Path to save test CSV')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max training steps')
    args = parser.parse_args()

    train_and_save(args.model_dir, args.input_csv, args.test_output_csv, args.max_steps)

