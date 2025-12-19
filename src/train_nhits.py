import os
import argparse
import pandas as pd
import numpy as np
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE, MQLoss
from pytorch_lightning.loggers import CSVLogger

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

def train_and_save(model_dir, input_path, test_output_path=None):
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
    # IMPORTANT: We include the lookback window (input_size=150) so the model has context
    # for the first prediction in the test set.
    # We also add a buffer (e.g. 20 steps) to allow for validation splits during evaluation
    # (since evaluate_nhits.py uses val_size=10, we need at least input_size + val_size history)
    input_size = 150
    buffer_size = 20
    start_idx = max(0, train_size + val_size - input_size - buffer_size)
    print(f"DEBUG: input_size={input_size}, buffer_size={buffer_size}, start_idx={start_idx}")
    test_df_export = Y_df.iloc[start_idx:]
    print(f"DEBUG: test_df_export length={len(test_df_export)}")
    
    if test_output_path:
        print(f"Saving test dataframe to {test_output_path}...")
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
        test_df_export.to_csv(test_output_path, index=False)
    
    # Define Exogenous Variables
    # Note: futr_exog_list requires these columns to be known in the future. 
    # Since they are calendar based, they are fine.
    futr_exog_list = ['temp', 'precip', 'snow', 'snowdepth', 'visibility', 'windspeed',
                      'week_sin', 'week_cos', 'day_sin', 'day_cos']
    hist_exog_list = ['rolling_mean_10', 'rolling_std_10', 'rolling_mean_50', 'rolling_std_50', 'rolling_max_10']
    
    # Setup Logger
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logger = CSVLogger(save_dir=model_dir, name="training_logs")

    # 2. Define Model
    models = [
        NHITS(
            h=1,                      # Horizon: Predict next step
            input_size=150,           # Lookback window (Aligned with GRU)
            loss=MQLoss(quantiles=[0.1, 0.5, 0.9]), 
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            max_steps=1000,           
            early_stop_patience_steps=10,
            val_check_steps=100,      # Check validation every 100 steps
            batch_size=256,           # Batch size (Aligned with GRU)
            scaler_type='robust',     
            learning_rate=1e-3,
            n_pool_kernel_size=[2, 2, 2], 
            n_freq_downsample=[168, 24, 1],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=logger      # Pass logger to capture history
        )
    ]
    
    # Initialize NeuralForecast
    # Assuming 'H' freq, or let it infer/use integer index if irregular
    # If data is irregular (train arrivals), 'H' might be wrong. 
    # But NHITS can handle integer index if we don't pass freq, or pass freq=1.
    # Let's try to infer or use a dummy freq if it's just sequence data.
    # For this dataset, it seems to be event-based (train arrivals), not strictly hourly.
    # So we might want to treat it as a sequence.
    nf = NeuralForecast(models=models, freq='H') # 'M' is minute? No, let's use default or 'auto'
    
    # 3. Train Model
    print("Training NHITS model...")
    # We pass only the train+val portion. 
    # val_size is the number of samples at the end of this dataframe to use for validation.
    nf.fit(df=train_val_df, val_size=val_size)
    
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
    parser.add_argument('--test_output_csv', type=str, required=False, help='Path to save test CSV')
    args = parser.parse_args()

    train_and_save(args.model_dir, args.input_csv, args.test_output_csv)

