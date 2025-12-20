import argparse
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import mae, rmse

def evaluate_nhits(model_dir, test_csv_path, metrics_output_path, plot_output_path, prediction_plot_path):
    print(f"Starting evaluation script...", flush=True)
    print(f"Loading test data from {test_csv_path}...", flush=True)
    test_df = pd.read_csv(test_csv_path)
    
    print(f"Test data columns: {test_df.columns.tolist()}", flush=True)
    print(f"Test data shape: {test_df.shape}", flush=True)
    
    # Ensure datetime
    if 'ds' in test_df.columns:
        test_df['ds'] = pd.to_datetime(test_df['ds'])
        # FIX: Ensure timezone-naive to avoid merge errors with NeuralForecast internal types
        if test_df['ds'].dt.tz is not None:
            test_df['ds'] = test_df['ds'].dt.tz_localize(None)
            
        # Do NOT round to hourly frequency. Use raw timestamps.
        # test_df['ds'] = test_df['ds'].dt.floor('H')
    
    # Ensure unique_id exists
    if 'unique_id' not in test_df.columns:
        test_df['unique_id'] = 'E'
        
    # Ensure data is sorted by time
    test_df = test_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
    # Aggregate duplicates if any exist after rounding (taking the mean)
    # if test_df.duplicated(subset=['unique_id', 'ds']).any():
    #    print("Aggregating duplicate timestamps after hourly rounding...")
    #    numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
    #    numeric_cols = [c for c in numeric_cols if c not in ['ds', 'unique_id']]
    #    test_df = test_df.groupby(['unique_id', 'ds'])[numeric_cols].mean().reset_index()

    print(f"Loading model from {model_dir}...", flush=True)
    try:
        nf = NeuralForecast.load(path=model_dir)
        print("Model loaded successfully.", flush=True)
            
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        raise e
    
    # Determine input size from the loaded model to calculate test size
    # nf.models is a list of models. We assume the first one.
    input_size = nf.models[0].input_size
    print(f"Model Input Size: {input_size}", flush=True)
    
    # Check for required columns (based on model configuration)
    # We can't easily inspect the model's exog list from here without digging into private attributes
    # but we can check if the dataframe looks reasonable.
    
    # Calculate the number of steps to predict
    # The test_df contains [lookback_window + actual_test_data]
    # We want to predict the actual_test_data
    # We must also account for the validation set (val_size=10) used in cross_validation
    # AND ensure we have strictly more than input_size for the training window
    
    # NOTE: train_nhits.py exports the file with a buffer of (input_size + 20) before the test set.
    # So the "Test Set" effectively starts after that buffer.
    # However, we don't know the exact buffer size here.
    # But we know we want to predict everything that is NOT the initial lookback window.
    # Let's assume we want to predict everything after the first input_size steps.
    
    # Actually, cross_validation takes 'test_size'.
    # We must ensure the remaining data (context) is large enough for the model to train (input_size + horizon).
    # We reserve input_size + 10 steps for context to avoid "No windows available" errors.
    
    # min_context = input_size + 10
    # n_test_steps = len(test_df) - min_context
    # print(f"Total rows in test_df: {len(test_df)}", flush=True)
    # print(f"Reserved Context: {min_context}", flush=True)
    # print(f"Expected test steps: {n_test_steps}", flush=True)
    
    # if n_test_steps <= 0:
    #     raise ValueError(f"Test dataframe is too small ({len(test_df)}) for input_size ({input_size}).")
        
    print(f"Forecasting using cross_validation...", flush=True)
    
    try:
        # Use user requested setup
        #subset_test_size = 1000 # Evaluate on first 1000 points of test set
        print(f"Running Cross Validation on subset ({len(test_df)} samples)...", flush=True)
        
        forecasts = nf.cross_validation(
            df=test_df,
            val_size=len(test_df),
            test_size=len(test_df),
            n_windows=None,
            step_size=1
        )
        print(f"Forecasts generated. Shape: {forecasts.shape}", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR during cross_validation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise e
    
    print("Forecasts generated. Columns:", forecasts.columns, flush=True)
    
    # Calculate Metrics manually as requested
    print("Calculating metrics...", flush=True)
    
    # Full Set Metrics
    y_true = forecasts['y']
    y_pred = forecasts['NHITS-median']
    
    # Use neuralforecast losses
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    
    print(f"MAE: {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")

    # Save Metrics (using Full Set results)
    metrics_list = [
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
    
    metrics = {"metrics": metrics_list}
    
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_output_path}")


def plot_predictions(forecasts_df, output_path, metrics_dict=None):
    # Plot a segment of the forecasts
    plot_df = forecasts_df.iloc[:200] # First 200 predictions

    plt.figure(figsize=(15, 5))
    plt.plot(plot_df['ds'], plot_df['y'], label='Actual MBT', color='black', alpha=0.7)
    
    # Check for median column
    y_pred_col = 'NHITS-median' if 'NHITS-median' in plot_df.columns else 'NHITS'
    plt.plot(plot_df['ds'], plot_df[y_pred_col], label='Predicted Median MBT', color='blue', linewidth=2)
    
    # Updated to use 80% prediction interval columns (derived from 0.1 and 0.9 quantiles)
    # Check for lo/hi columns
    lo_col = 'NHITS-lo-80.0' if 'NHITS-lo-80.0' in plot_df.columns else 'NHITS-lo-80'
    hi_col = 'NHITS-hi-80.0' if 'NHITS-hi-80.0' in plot_df.columns else 'NHITS-hi-80'
    
    if lo_col in plot_df.columns and hi_col in plot_df.columns:
        plt.fill_between(plot_df['ds'], plot_df[lo_col], plot_df[hi_col], color='blue', alpha=0.2, label='80% Confidence Interval')
        
    plt.title('Subway Headway Prediction: Actual vs Predicted (with Uncertainty)')
    plt.xlabel('Time')
    plt.ylabel('Minutes Between Trains (MBT)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure using the helper function to embed in HTML
    save_plot_html(output_path, "Subway Headway Prediction", metrics_dict)

def save_plot_html(output_path, title, metrics_dict=None):
    import base64
    from io import BytesIO
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    metrics_html = ""
    if metrics_dict:
        rows = ""
        for k, v in metrics_dict.items():
            rows += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        metrics_html = f"""
        <div style="margin-bottom: 20px;">
            <h3>Metrics</h3>
            <table border="1" style="border-collapse: collapse; width: 300px;">
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 8px; text-align: left;">Metric</th>
                    <th style="padding: 8px; text-align: left;">Value</th>
                </tr>
                {rows}
            </table>
        </div>
        """
    
    html_content = f"""
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border: 1px solid #ddd; }}
            th, td {{ text-align: left; padding: 8px; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        {metrics_html}
        <img src="data:image/png;base64,{img_base64}" alt="{title}" style="max-width: 100%; border: 1px solid #ddd;">
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