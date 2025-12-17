import argparse
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import mae, rmse

def evaluate_nhits(model_dir, test_csv_path, metrics_output_path, plot_output_path, prediction_plot_path):
    print(f"Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    
    # Ensure datetime
    if 'ds' in test_df.columns:
        test_df['ds'] = pd.to_datetime(test_df['ds'])
    
    # Ensure unique_id exists
    if 'unique_id' not in test_df.columns:
        test_df['unique_id'] = 'E'

    print(f"Loading model from {model_dir}...")
    nf = NeuralForecast.load(path=model_dir)
    
    # Determine input size from the loaded model to calculate test size
    # nf.models is a list of models. We assume the first one.
    input_size = nf.models[0].input_size
    print(f"Model Input Size: {input_size}")
    
    # Calculate the number of steps to predict
    # The test_df contains [lookback_window + actual_test_data]
    # We want to predict the actual_test_data
    n_test_steps = len(test_df) - input_size
    
    if n_test_steps <= 0:
        raise ValueError(f"Test dataframe is too small ({len(test_df)}) for input_size ({input_size}).")
        
    print(f"Forecasting {n_test_steps} steps using cross_validation (step_size=1)...")
    
    # Use cross_validation to predict step-by-step
    # This simulates a production scenario where we predict the next step, then observe the actual, then predict again.
    forecasts = nf.cross_validation(
        df=test_df,
        val_size=0, # We don't need a validation split here, just test
        test_size=n_test_steps,
        n_windows=None,
        step_size=1
    )
    
    print("Forecasts generated. Columns:", forecasts.columns)
    
    # Calculate Metrics
    y_true = forecasts['y']
    # Use median prediction if available (MQLoss), else 'NHITS'
    y_pred_col = 'NHITS-median' if 'NHITS-median' in forecasts.columns else 'NHITS'
    y_pred = forecasts[y_pred_col]
    
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    
    print(f"Test MAE: {mae_val}")
    print(f"Test RMSE: {rmse_val}")
    
    # Save Metrics
    metrics = {
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
    
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_output_path}")

    # Plotting
    if plot_output_path:
        plot_loss(model_dir, plot_output_path)
        
    if prediction_plot_path:
        plot_predictions(forecasts, prediction_plot_path)

def plot_loss(model_dir, output_path):
    logs_dir = os.path.join(model_dir, "training_logs")
    if not os.path.exists(logs_dir):
        print("No training logs found.")
        return
        
    versions = [d for d in os.listdir(logs_dir) if d.startswith("version_")]
    if not versions:
        print("No version directory found in logs.")
        return
        
    # Sort versions to get the latest
    versions.sort()
    metrics_path = os.path.join(logs_dir, versions[-1], "metrics.csv")
    
    if not os.path.exists(metrics_path):
        print(f"No metrics.csv found at {metrics_path}")
        return
        
    print(f"Loading metrics from {metrics_path}...")
    metrics_df = pd.read_csv(metrics_path)
    
    plt.figure(figsize=(10, 6))
    
    # Check for column names (PyTorch Lightning logger)
    if 'train_loss_epoch' in metrics_df.columns:
        plt.plot(metrics_df['train_loss_epoch'].dropna(), label='Train Loss')
    elif 'train_loss' in metrics_df.columns:
        plt.plot(metrics_df['train_loss'].dropna(), label='Train Loss')
        
    if 'valid_loss' in metrics_df.columns:
        plt.plot(metrics_df['valid_loss'].dropna(), label='Validation Loss')
        
    plt.title('N-HiTS Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_plot_html(output_path, "Training Loss")

def plot_predictions(forecasts_df, output_path):
    plt.figure(figsize=(15, 6))
    
    # Plot a subset if too large
    limit = 500
    plot_df = forecasts_df.iloc[:limit]
    
    plt.plot(plot_df['ds'], plot_df['y'], label='Actual MBT', color='black', alpha=0.7)
    
    # Check for median or default
    y_pred_col = 'NHITS-median' if 'NHITS-median' in plot_df.columns else 'NHITS'
    plt.plot(plot_df['ds'], plot_df[y_pred_col], label='Predicted Median MBT', color='blue', linewidth=2)
    
    # Plot Confidence Intervals if available
    if 'NHITS-lo-80.0' in plot_df.columns and 'NHITS-hi-80.0' in plot_df.columns:
        plt.fill_between(
            plot_df['ds'], 
            plot_df['NHITS-lo-80.0'], 
            plot_df['NHITS-hi-80.0'], 
            color='blue', 
            alpha=0.2, 
            label='80% Confidence Interval'
        )
        
    plt.title(f'Subway Headway Prediction: Actual vs Predicted (First {limit} samples)')
    plt.xlabel('Time')
    plt.ylabel('Minutes Between Trains (MBT)')
    plt.legend()
    plt.grid(True)
    
    save_plot_html(output_path, "Prediction Plot")

def save_plot_html(output_path, title):
    import base64
    from io import BytesIO
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    html_content = f"""
    <html>
    <head><title>{title}</title></head>
    <body>
        <h1>{title}</h1>
        <img src="data:image/png;base64,{img_base64}" alt="{title}">
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--plot_output_path', type=str, required=False)
    parser.add_argument('--prediction_plot_path', type=str, required=False)
    
    args = parser.parse_args()
    
    evaluate_nhits(
        args.model_dir, 
        args.test_dataset_path, 
        args.metrics_output_path, 
        args.plot_output_path, 
        args.prediction_plot_path
    )