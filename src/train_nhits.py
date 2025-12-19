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
        
    # Round to hourly frequency to ensure alignment with NeuralForecast's freq='H'
    # This fixes potential mismatches during cross_validation merge
    df['ds'] = df['ds'].dt.floor('H')
    
    # Add unique_id
    df['unique_id'] = 'E'
    
    # Aggregate duplicates if any exist after rounding (taking the mean)
    # This ensures strictly one record per hour per unique_id
    if df.duplicated(subset=['unique_id', 'ds']).any():
        print("Aggregating duplicate timestamps after hourly rounding...")
        # Identify numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Ensure 'ds' and 'unique_id' are not in numeric_cols to avoid errors, though groupby handles them
        numeric_cols = [c for c in numeric_cols if c not in ['ds', 'unique_id']]
        
        df = df.groupby(['unique_id', 'ds'])[numeric_cols].mean().reset_index()
    
    print(f"Data shape after processing: {df.shape}")
    return df

def train_and_save(model_dir, input_path, test_output_path=None, max_steps=1000, metrics_output_path=None, plot_output_path=None):
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
    # We also add a buffer (e.g. 20 steps) to allow for validation splits during evaluation
    # (since evaluate_nhits.py uses val_size=10, we need at least input_size + val_size history)
    input_size = 160
    buffer_size = 20
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
            scaler_type='robust',     
            learning_rate=1e-3,
            n_pool_kernel_size=[2, 2, 2], 
            n_freq_downsample=[168, 24, 1],
            accelerator="gpu" if torch.cuda.is_available() else "cpu"
            # logger=logger      # Removed to prevent path issues during loading
        )
    ]
    
    # Initialize NeuralForecast
    nf = NeuralForecast(models=models, freq='H')
    
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

    # 5. Forecasting and Evaluation (Integrated)
    print("Starting in-script evaluation...")

    # Use the full dataframe (Y_df) for cross_validation to allow it to see history
    # We want to evaluate on the test set.
    # The user code suggests subsetting for speed.
    
    # Let's use the test_size calculated earlier, but cap it if needed for speed
    # User requested subset_test_size = 1000
    subset_test_size = min(1000, len(test_df))
    print(f"Evaluating on last {subset_test_size} steps...")
    
    # Using cross_validation to simulate production behavior on the test set
    forecasts = nf.cross_validation(
        df=Y_df,
        val_size=subset_test_size,
        test_size=subset_test_size,
        n_windows=None, # Automatically determined by val_size/test_size logic if not set
        step_size=1     # Predict 1 step at a time
    )

    # Calculate Metrics
    y_true = forecasts['y']
    
    # Determine prediction column
    if 'NHITS-median' in forecasts.columns:
        y_pred = forecasts['NHITS-median']
    elif 'NHITS' in forecasts.columns:
        y_pred = forecasts['NHITS']
    else:
        # Fallback to first prediction column if specific names aren't found
        pred_cols = [c for c in forecasts.columns if c not in ['unique_id', 'ds', 'cutoff', 'y']]
        if pred_cols:
            y_pred = forecasts[pred_cols[0]]
        else:
            raise ValueError("No prediction columns found in forecasts")

    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    print(f"MAE: {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")

    if metrics_output_path:
        metrics_data = {
            "metrics": [
                {
                    "name": "mae",
                    "numberValue": float(mae_val),
                    "format": "RAW"
                },
                {
                    "name": "rmse",
                    "numberValue": float(rmse_val),
                    "format": "RAW"
                }
            ]
        }
        print(f"Saving metrics to {metrics_output_path}...")
        os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics_data, f)

    ## 9. Visualizing Predictions vs Actuals

    # Plot a segment of the forecasts
    plot_df = forecasts.iloc[:200].copy() # First 200 predictions
    
    # Debug: Check if actuals are present
    if plot_df['y'].isnull().all():
        print("WARNING: 'y' (Actuals) column in plot dataframe is entirely NaN. Check timestamp alignment.")
    else:
        print(f"Plotting {plot_df['y'].count()} actual values against predictions.")

    plt.figure(figsize=(15, 5))
    plt.plot(plot_df['ds'], plot_df['y'], label='Actual MBT', color='black', alpha=0.7)
    
    if 'NHITS-median' in plot_df.columns:
        plt.plot(plot_df['ds'], plot_df['NHITS-median'], label='Predicted Median MBT', color='blue', linewidth=2)
    elif 'NHITS' in plot_df.columns:
        plt.plot(plot_df['ds'], plot_df['NHITS'], label='Predicted MBT', color='blue', linewidth=2)
        
    # Updated to use 80% prediction interval columns (derived from 0.1 and 0.9 quantiles)
    lo_col = None
    hi_col = None
    
    if 'NHITS-lo-80.0' in plot_df.columns: lo_col = 'NHITS-lo-80.0'
    elif 'NHITS-lo-80' in plot_df.columns: lo_col = 'NHITS-lo-80'
    
    if 'NHITS-hi-80.0' in plot_df.columns: hi_col = 'NHITS-hi-80.0'
    elif 'NHITS-hi-80' in plot_df.columns: hi_col = 'NHITS-hi-80'

    if lo_col and hi_col:
        plt.fill_between(plot_df['ds'], plot_df[lo_col], plot_df[hi_col], color='blue', alpha=0.2, label='80% Confidence Interval')
        
    plt.title('Subway Headway Prediction: Actual vs Predicted (with Uncertainty)')
    plt.xlabel('Time')
    plt.ylabel('Minutes Between Trains (MBT)')
    plt.legend()

    # Save the figure
    plot_output_path_png = os.path.join(model_dir, 'subway_headway_forecast.png')
    plt.savefig(plot_output_path_png, bbox_inches='tight')
    print(f"Plot saved as '{plot_output_path_png}'")

    if plot_output_path:
        # Save as HTML for KFP visualization
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        html_content = f"""
        <html>
        <head><title>N-HiTS Forecast</title></head>
        <body>
            <h1>N-HiTS Forecast vs Actuals</h1>
            <img src="data:image/png;base64,{img_base64}" alt="Forecast Plot">
        </body>
        </html>
        """
        
        print(f"Saving plot HTML to {plot_output_path}...")
        os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
        with open(plot_output_path, 'w') as f:
            f.write(html_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model_dir', type=str, default='nhits_model', help='Local directory to save model')
    parser.add_argument('--test_output_csv', type=str, required=False, help='Path to save test CSV')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max training steps')
    parser.add_argument('--metrics_output_path', type=str, required=False, help='Path to save metrics JSON')
    parser.add_argument('--plot_output_path', type=str, required=False, help='Path to save plot HTML')
    args = parser.parse_args()

    train_and_save(args.model_dir, args.input_csv, args.test_output_csv, args.max_steps, args.metrics_output_path, args.plot_output_path)

