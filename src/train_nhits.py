import os
import argparse
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE, MQLoss

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
    
    # Add unique_id
    df['unique_id'] = 'E'
    
    print(f"Data shape after processing: {df.shape}")
    return df

def train_and_save(model_dir, input_path):
    # 1. Load Data
    Y_df = load_data(input_path)
    
    # Split logic (60/20/20)
    n = len(Y_df)
    val_size = int(n * 0.2)
    
    # Define Exogenous Variables
    # Note: futr_exog_list requires these columns to be known in the future. 
    # Since they are calendar based, they are fine.
    futr_exog_list = ['temp', 'precip', 'snow', 'snowdepth', 'visibility', 'windspeed',
                      'week_sin', 'week_cos', 'day_sin', 'day_cos']
    hist_exog_list = ['rolling_mean_10', 'rolling_std_10', 'rolling_mean_50', 'rolling_std_50', 'rolling_max_10']
    
    # 2. Define Model
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
            n_freq_downsample=[168, 24, 1],
            accelerator="cpu" # Use CPU for now to match container, or "gpu" if available
        )
    ]
    
    # Initialize NeuralForecast
    # Assuming 'H' freq, or let it infer/use integer index if irregular
    # If data is irregular (train arrivals), 'H' might be wrong. 
    # But NHITS can handle integer index if we don't pass freq, or pass freq=1.
    # Let's try to infer or use a dummy freq if it's just sequence data.
    # For this dataset, it seems to be event-based (train arrivals), not strictly hourly.
    # So we might want to treat it as a sequence.
    nf = NeuralForecast(models=models, freq='M') # 'M' is minute? No, let's use default or 'auto'
    
    # 3. Train Model
    print("Training NHITS model...")
    nf.fit(df=Y_df, val_size=val_size)
    
    # 4. Save Model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    print(f"Saving model to {model_dir}...")
    nf.save(path=model_dir, model_index=None, overwrite=True)
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model_dir', type=str, default='nhits_model', help='Local directory to save model')
    args = parser.parse_args()

    train_and_save(args.model_dir, args.input_csv)

