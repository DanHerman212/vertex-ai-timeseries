import os
import argparse
import pandas as pd
import numpy as np
import torch
import json
import pickle
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

def train_and_save(model_dir, input_path, df_output_path=None, logs_dir=None):
    # 1. Load Data
    df = load_data(input_path)
    
    # # Split logic (60/20/20)
    # n = len(df)
    # train_size = int(n * 0.6)
    # val_size = int(n * 0.2)
    # test_size = n - train_size - val_size

    # train_df = df.iloc[:train_size]
    # val_df = df.iloc[train_size:train_size+val_size]
    # test_df = df.iloc[train_size+val_size:]

    # print(f"Train size: {len(train_df)}")
    # print(f"Validation size: {len(val_df)}")
    # print(f"Test size: {len(test_df)}")
    
    
    input_size = 160
    horizon = 1
    
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
            h=horizon,                      # Horizon: Predict next step
            input_size=input_size,           # Lookback window (Aligned with Notebook)
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), 
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            max_steps=1000,           
            early_stop_patience_steps=10,
            scaler_type='standard', 
            learning_rate=1e-3,
            n_pool_kernel_size=[2, 2, 2], 
            n_freq_downsample=[168, 24, 1],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=logger
        )
    ]
    
    # Initialize NeuralForecast
    # Use freq='H' to match the notebook configuration, even with sub-hourly data
    nf = NeuralForecast(models=models, freq='H')
    
    # 3. Train Model
    print("Training NHITS model...")
    nf.fit(df=df, val_size=horizon)
    
    # 4. Save Model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print(f"Saving model to {model_dir}...")
    nf.save(path=model_dir, model_index=None, overwrite=True)

    # Verify artifacts
    print(f"Verifying artifacts in {model_dir}...")
    try:
        saved_files = os.listdir(model_dir)
        print(f"Files in model_dir: {saved_files}")
        
        if 'dataset.pkl' not in saved_files:
            print("WARNING: dataset.pkl not found. Saving manually.")
            with open(os.path.join(model_dir, 'dataset.pkl'), 'wb') as f:
                pickle.dump(nf.dataset, f)
            print("Manually saved dataset.pkl")
    except Exception as e:
        print(f"Error verifying/saving artifacts: {e}")
    
    # 5. Save Full Data
    if df_output_path:
        print(f"Saving full dataframe to {df_output_path}...")
        os.makedirs(os.path.dirname(df_output_path), exist_ok=True)
        df.to_csv(df_output_path, index=False)

    # 6. Save Training Logs
    # Copy from temp dir to model_dir/training_logs
    final_log_dir = os.path.join(model_dir, "training_logs")
    if os.path.exists(final_log_dir):
        shutil.rmtree(final_log_dir)
    shutil.copytree(temp_log_dir, final_log_dir)
    print(f"Training logs saved to {final_log_dir}")

    # Save logs to separate artifact if requested
    if logs_dir:
        print(f"Saving training logs to {logs_dir}...")
        # KFP creates the directory for the artifact, but copytree expects destination to NOT exist or be empty?
        # shutil.copytree requires dst to not exist.
        # But KFP might create the folder.
        # If logs_dir exists, we should copy the CONTENTS of temp_log_dir to logs_dir.
        if os.path.exists(logs_dir):
             # Copy contents using shutil with dirs_exist_ok=True (Python 3.8+)
             shutil.copytree(temp_log_dir, logs_dir, dirs_exist_ok=True)
        else:
             shutil.copytree(temp_log_dir, logs_dir)
        print(f"Training logs saved to {logs_dir}")
    
    # Cleanup temp logs
    if os.path.exists(temp_log_dir):
        shutil.rmtree(temp_log_dir)
    print("Model and artifacts saved successfully.")

if __name__ == "__main__":
    print("Starting train_nhits.py...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model_dir', type=str, default='nhits_model', help='Local directory to save model')
    parser.add_argument('--df_output_csv', type=str, required=False, help='Path to save full df to CSV')
    parser.add_argument('--logs_dir', type=str, required=False, help='Path to save training logs')
    
    try:
        args = parser.parse_args()
        print(f"Arguments: {args}")
        train_and_save(args.model_dir, args.input_csv, args.df_output_csv, args.logs_dir)
    except Exception as e:
        print(f"Error in train_nhits.py: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

