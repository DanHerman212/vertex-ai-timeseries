import sys
print("Starting evaluate_nhits.py script...", flush=True)

try:
    import argparse
    import pandas as pd
    import numpy as np
    import os
    import json
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.numpy import mae, rmse
    from pytorch_lightning.loggers import CSVLogger
    import tempfile
    print("All imports successful.", flush=True)
except ImportError as e:
    print(f"CRITICAL: Import failed: {e}", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: Unexpected error during imports: {e}", flush=True)
    sys.exit(1)

def get_plot_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_loss(logs_dir):
    print(f"Plotting loss from logs in {logs_dir}...", flush=True)
    # Expected path: logs_dir/training_logs/version_0/metrics.csv
    # Note: The structure depends on how it was copied. 
    # In train_nhits.py: shutil.copytree(temp_log_dir, logs_dir)
    # temp_log_dir contained 'training_logs/version_0/metrics.csv' because name="training_logs"
    
    # Let's try to find metrics.csv recursively
    metrics_path = None
    for root, dirs, files in os.walk(logs_dir):
        if "metrics.csv" in files:
            metrics_path = os.path.join(root, "metrics.csv")
            break
            
    if not metrics_path:
        print("Warning: metrics.csv not found in logs_dir. Skipping loss plot.", flush=True)
        return None

    try:
        df = pd.read_csv(metrics_path)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by epoch to get one value per epoch if multiple steps
        # Or just plot by step
        if 'step' in df.columns:
            x_axis = 'step'
        else:
            x_axis = df.index
            
        if 'train_loss' in df.columns:
            # Filter out NaNs which might happen if validation is on different steps
            train_df = df.dropna(subset=['train_loss'])
            ax.plot(train_df[x_axis], train_df['train_loss'], label='Train Loss')
            
        if 'valid_loss' in df.columns:
            val_df = df.dropna(subset=['valid_loss'])
            ax.plot(val_df[x_axis], val_df['valid_loss'], label='Validation Loss')
            
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        img_base64 = get_plot_base64(fig)
        plt.close(fig)
        return img_base64
        
    except Exception as e:
        print(f"Error plotting loss: {e}", flush=True)
        return None

def plot_predictions(forecasts_df):
    # Plot a segment of the forecasts
    plot_df = forecasts_df.iloc[:200] # First 200 predictions

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(plot_df['ds'], plot_df['y'], label='Actual MBT', color='black', alpha=0.7)
    
    # Identify Median Column
    median_col = None
    if 'NHITS-median' in plot_df.columns:
        median_col = 'NHITS-median'
    elif 'NHITS-q-0.5' in plot_df.columns:
        median_col = 'NHITS-q-0.5'
    elif 'NHITS' in plot_df.columns:
        median_col = 'NHITS'
        
    if median_col:
        ax.plot(plot_df['ds'], plot_df[median_col], label='Predicted Median MBT', color='blue', linewidth=2)
    
    # Identify Quantile Columns for Confidence Interval
    # We expect q-0.1 and q-0.9 from MQLoss(quantiles=[0.1, 0.5, 0.9])
    lo_col = None
    hi_col = None
    
    if 'NHITS-q-0.1' in plot_df.columns and 'NHITS-q-0.9' in plot_df.columns:
        lo_col = 'NHITS-q-0.1'
        hi_col = 'NHITS-q-0.9'
        label = '10%-90% Quantile Range'
    elif 'NHITS-lo-90.0' in plot_df.columns and 'NHITS-hi-90.0' in plot_df.columns:
         lo_col = 'NHITS-lo-90.0'
         hi_col = 'NHITS-hi-90.0'
         label = '90% Confidence Interval'
        
    if lo_col and hi_col:
        ax.fill_between(plot_df['ds'], plot_df[lo_col], plot_df[hi_col], color='blue', alpha=0.2, label=label)
        
    ax.set_title('Subway Headway Prediction: Actual vs Predicted')
    ax.set_xlabel('Time')
    ax.set_ylabel('Minutes Between Trains (MBT)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    img_base64 = get_plot_base64(fig)
    plt.close(fig)
    return img_base64

def generate_html_report(output_path, metrics_dict, pred_plot_b64, loss_plot_b64):
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
    
    loss_html = ""
    if loss_plot_b64:
        loss_html = f"""
        <div style="margin-bottom: 40px;">
            <h3>Training & Validation Loss</h3>
            <img src="data:image/png;base64,{loss_plot_b64}" alt="Loss Plot" style="max-width: 100%; border: 1px solid #ddd;">
        </div>
        """
        
    pred_html = ""
    if pred_plot_b64:
        pred_html = f"""
        <div style="margin-bottom: 40px;">
            <h3>Forecast Visualization</h3>
            <img src="data:image/png;base64,{pred_plot_b64}" alt="Prediction Plot" style="max-width: 100%; border: 1px solid #ddd;">
        </div>
        """

    html_content = f"""
    <html>
    <head>
        <title>NHITS Model Evaluation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border: 1px solid #ddd; }}
            th, td {{ text-align: left; padding: 8px; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            h1, h3 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>NHITS Model Evaluation Report</h1>
        {metrics_html}
        {loss_html}
        {pred_html}
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"HTML report saved to {output_path}", flush=True)

def evaluate_nhits(model_dir, df_csv_path, metrics_output_path, html_output_path, logs_dir=None):
    print(f"Starting evaluation script...", flush=True)
    print(f"Loading full data from {df_csv_path}...", flush=True)
    df = pd.read_csv(df_csv_path)
    
    print(f"data columns: {df.columns.tolist()}", flush=True)
    print(f"data shape: {df.shape}", flush=True)
    
    # Ensure datetime
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
    
    # Ensure unique_id exists
    if 'unique_id' not in df.columns:
        df['unique_id'] = 'E'
        
    # Ensure data is sorted by time
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        

    print(f"Loading model from {model_dir}...", flush=True)
    try:
        nf = NeuralForecast.load(path=model_dir)
        print("Model loaded successfully.", flush=True)
        
        # FIX: Disable logger completely to avoid "Missing folder" error
        # We don't need training logs during evaluation/inference.
        for model in nf.models:
            # Disable logger in trainer_kwargs
            if hasattr(model, 'trainer_kwargs') and isinstance(model.trainer_kwargs, dict):
                model.trainer_kwargs['logger'] = False
                model.trainer_kwargs['enable_checkpointing'] = False

            # Clear existing logger instance
            if hasattr(model, '_logger'):
                model._logger = None
            
            # Clear trainer reference so it re-initializes with new kwargs
            if hasattr(model, '_trainer'):
                model._trainer = None
                
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        raise e
    
    try:
        # ---------------------------------------------------------
        # ROLLING FORECAST LOOP (Inference Mode)
        # ---------------------------------------------------------
        # We want to evaluate on the last 20% of the data (Test Set).
        # To predict the first point of the test set, we need the history 
        # immediately preceding it.
        
        total_rows = len(df)
        test_size = int(total_rows * 0.2)
        test_start_idx = total_rows - test_size
        
        print(f"Total rows: {total_rows}", flush=True)
        print(f"Test set size (last 20%): {test_size}", flush=True)
        print(f"Test start index: {test_start_idx}", flush=True)
        
        # We need to know the model's input size to keep the buffer efficient
        # Assuming the first model in the list is the one we use
        # (NeuralForecast can hold multiple, but we trained one NHITS)
        model_input_size = nf.models[0].input_size
        print(f"Model input size: {model_input_size}", flush=True)
        
        # Buffer to hold recent history. 
        # We start with enough history to cover the input_size before the test set.
        # We take a bit more to be safe.
        buffer_start_idx = max(0, test_start_idx - model_input_size - 10)
        history_buffer = df.iloc[buffer_start_idx:test_start_idx].copy()
        
        test_df = df.iloc[test_start_idx:].copy()
        
        predictions = []
        actuals = []
        timestamps = []
        
        print("Starting rolling forecast loop...", flush=True)
        
        # Iterate through the test set
        # For each step:
        # 1. Predict next step using history_buffer
        # 2. Store prediction
        # 3. Add actual observation to history_buffer for next step
        
        # Optimization: nf.predict is relatively heavy. 
        # If test set is large, this loop might take time.
        # But for ~400-2000 points it should be acceptable.
        
        for idx, row in test_df.iterrows():
            # Predict
            # nf.predict uses the end of the dataframe to predict forward
            # Since we have future exogenous variables, we must provide them in futr_df
            # We need the future exogenous variables for the NEXT step (which is 'row')
            
            # Construct futr_df for the next step
            # It needs 'unique_id', 'ds' and the future exogenous columns
            # The 'ds' should be the timestamp we are predicting (which is row['ds'])
            
            # Identify future exogenous columns from the model
            # We can infer them from the error message or model config, but let's assume we know them
            # or just pass all columns except 'y'
            
            futr_exog_cols = ['temp', 'precip', 'snow', 'snowdepth', 'visibility', 'windspeed', 'dow']
            
            # Create a single-row dataframe for the future step
            # futr_df = pd.DataFrame([row])
            # Ensure it has the right columns
            # futr_df = futr_df[['unique_id', 'ds'] + futr_exog_cols]
            
            # ROBUST FIX: Use make_future_dataframe to get the exact timestamp the model expects
            # This handles frequency alignment automatically.
            futr_df = nf.make_future_dataframe(df=history_buffer)
            
            # Now we need to populate the exogenous columns in this futr_df
            # We can look up the values from our original 'df' based on the timestamp 'ds'
            # This is efficient enough for evaluation
            
            # Merge with original df to get exog values
            # We use 'ds' and 'unique_id' as keys
            futr_df = futr_df.drop(columns=futr_exog_cols, errors='ignore') # Drop if they exist empty
            futr_df = futr_df.merge(df[['unique_id', 'ds'] + futr_exog_cols], on=['unique_id', 'ds'], how='left')
            
            # Handle potential missing values if the model predicts a time not in our test set (e.g. gap filling)
            # For now, forward fill or fill with 0
            futr_df = futr_df.fillna(method='ffill').fillna(0)
            
            fcst_df = nf.predict(df=history_buffer, futr_df=futr_df)
            
            # Extract prediction (NHITS-median usually, or just NHITS)
            # The column name depends on the model alias. 
            # Default alias for NHITS is 'NHITS'. 
            # If loss was MQLoss, we might have quantiles.
            
            # Let's find the prediction column
            pred_cols = [c for c in fcst_df.columns if c not in ['ds', 'unique_id']]
            # Prefer median or main point forecast
            pred_col = next((c for c in pred_cols if 'median' in c), pred_cols[0])
            
            pred_value = fcst_df.iloc[0][pred_col]
            
            predictions.append(pred_value)
            actuals.append(row['y'])
            timestamps.append(row['ds'])
            
            # Update buffer: drop oldest, add new actual
            # We append the current test row to history so it's available for the NEXT prediction
            history_buffer = pd.concat([history_buffer.iloc[1:], df.iloc[[idx]]])
            
            if len(predictions) % 50 == 0:
                print(f"Processed {len(predictions)}/{test_size} steps...", flush=True)

        # Construct Forecast DataFrame matching the structure expected by plotting functions
        forecasts = pd.DataFrame({
            'ds': timestamps,
            'y': actuals,
            'NHITS-median': predictions
        })
        
        # Add dummy quantile columns if plotting expects them, to avoid errors
        # (Or update plotting logic. For now, we just plot median)
        forecasts['unique_id'] = 'E'
        
        print(f"Forecasts generated. Shape: {forecasts.shape}", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR during rolling forecast: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise e
    
    print("Forecasts generated. Columns:", forecasts.columns, flush=True)
    
    # Calculate Metrics manually as requested
    print("Calculating metrics...", flush=True)
    
    # Full Set Metrics
    y_true = forecasts['y']
    
    # Identify Median Column for Metrics
    median_col = None
    if 'NHITS-median' in forecasts.columns:
        median_col = 'NHITS-median'
    elif 'NHITS-q-0.5' in forecasts.columns:
        median_col = 'NHITS-q-0.5'
    elif 'NHITS' in forecasts.columns:
        median_col = 'NHITS'
        
    if median_col:
        y_pred = forecasts[median_col]
        # Use neuralforecast losses
        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
    else:
        print("Warning: Could not find median column for metrics calculation. Using 0.", flush=True)
        mae_val = 0.0
        rmse_val = 0.0
    
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

    # Generate Plots and HTML Report
    print("Generating plots...", flush=True)
    
    # 1. Prediction Plot
    pred_plot_b64 = plot_predictions(forecasts)
    
    # 2. Loss Plot (if logs available)
    loss_plot_b64 = None
    if logs_dir:
        loss_plot_b64 = plot_loss(logs_dir)
        
    # 3. Generate HTML
    metrics_dict = {"MAE": mae_val, "RMSE": rmse_val}
    generate_html_report(html_output_path, metrics_dict, pred_plot_b64, loss_plot_b64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_csv_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--html_output_path', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=False)
    
    args = parser.parse_args()
    
    evaluate_nhits(
        args.model_dir, 
        args.df_csv_path, 
        args.metrics_output_path, 
        args.html_output_path,
        args.logs_dir
    )