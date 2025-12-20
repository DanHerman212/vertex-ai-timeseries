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
# from neuralforecast.losses.numpy import mae, rmse # Replaced by utilsforecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, scaled_crps

def evaluate_nhits(model_dir, test_csv_path, metrics_output_path, plot_output_path, prediction_plot_path):
    print(f"Starting evaluation script...", flush=True)
    print(f"Loading test data from {test_csv_path}...", flush=True)
    test_df = pd.read_csv(test_csv_path)
    
    print(f"Test data columns: {test_df.columns.tolist()}", flush=True)
    
    # Ensure datetime
    if 'ds' in test_df.columns:
        test_df['ds'] = pd.to_datetime(test_df['ds'])
        # FIX: Ensure timezone-naive to avoid merge errors with NeuralForecast internal types
        if test_df['ds'].dt.tz is not None:
            test_df['ds'] = test_df['ds'].dt.tz_localize(None)
            
        # Round to hourly frequency to ensure alignment with NeuralForecast's freq='H'
        test_df['ds'] = test_df['ds'].dt.floor('H')
    
    # Ensure unique_id exists
    if 'unique_id' not in test_df.columns:
        test_df['unique_id'] = 'E'
        
    # Aggregate duplicates if any exist after rounding (taking the mean)
    if test_df.duplicated(subset=['unique_id', 'ds']).any():
        print("Aggregating duplicate timestamps after hourly rounding...")
        numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['ds', 'unique_id']]
        test_df = test_df.groupby(['unique_id', 'ds'])[numeric_cols].mean().reset_index()

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
    
    min_context = input_size + 10
    n_test_steps = len(test_df) - min_context
    print(f"Total rows in test_df: {len(test_df)}", flush=True)
    print(f"Reserved Context: {min_context}", flush=True)
    print(f"Expected test steps: {n_test_steps}", flush=True)
    
    if n_test_steps <= 0:
        raise ValueError(f"Test dataframe is too small ({len(test_df)}) for input_size ({input_size}).")
        
    print(f"Forecasting {n_test_steps} steps using predict...", flush=True)
    
    try:
        # Split data into history (context) and future (ground truth)
        # We use the loaded model weights (no refit) to predict the test period
        history_df = test_df.iloc[:-n_test_steps]
        future_df = test_df.iloc[-n_test_steps:]
        
        print(f"History size: {len(history_df)}, Future size: {len(future_df)}", flush=True)
        
        forecasts = nf.predict(df=history_df, h=n_test_steps)
        
        # Merge actuals (y) from future_df into forecasts for metric calculation
        # forecasts: [unique_id, ds, NHITS, ...]
        # future_df: [unique_id, ds, y, ...]
        forecasts = forecasts.merge(future_df[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')
            
    except Exception as e:
        print(f"CRITICAL ERROR during predict_insample: {e}", flush=True)
        # Try to print more context
        import traceback
        traceback.print_exc()
        raise e
    
    print("Forecasts generated. Columns:", forecasts.columns, flush=True)
    
    # Calculate Metrics using utilsforecast
    # Prepare dataframe: Rename NHITS-median to NHITS for standard evaluation
    eval_df = forecasts.copy()
    if 'NHITS-median' in eval_df.columns:
        eval_df = eval_df.rename(columns={'NHITS-median': 'NHITS'})
        
    # Drop cutoff column if present, otherwise utilsforecast calculates metrics per cutoff (window)
    # resulting in MAE == RMSE for step_size=1
    if 'cutoff' in eval_df.columns:
        eval_df = eval_df.drop(columns=['cutoff'])

    # Fix column names for quantiles (remove .0 suffix)
    # utilsforecast expects 'NHITS-lo-80', but NeuralForecast outputs 'NHITS-lo-80.0'
    rename_dict = {}
    for col in eval_df.columns:
        if col.endswith('.0'):
            new_col = col.replace('.0', '')
            rename_dict[col] = new_col
            
    if rename_dict:
        print(f"Renaming columns for utilsforecast compatibility: {rename_dict}", flush=True)
        eval_df = eval_df.rename(columns=rename_dict)
        
    # Define metrics
    metrics = [mae, rmse, scaled_crps]
    
    try:
        print("Calculating metrics using utilsforecast...", flush=True)
        # We pass level=[80] because our model was trained with quantiles 0.1, 0.5, 0.9
        # which corresponds to an 80% prediction interval (10% to 90%).
        # The columns NHITS-lo-80.0 and NHITS-hi-80.0 should exist.
        evaluation_df = evaluate(
            eval_df,
            metrics=metrics,
            models=['NHITS'],
            target_col='y',
            id_col='unique_id',
            time_col='ds',
            level=[80]
        )
        
        print("Evaluation results:", evaluation_df, flush=True)
        
        # Aggregate results (mean over unique_ids)
        summary = evaluation_df.drop(columns=['unique_id']).groupby('metric').mean().reset_index()
        
        mae_val = summary.loc[summary['metric'] == 'mae', 'NHITS'].values[0]
        rmse_val = summary.loc[summary['metric'] == 'rmse', 'NHITS'].values[0]
        crps_val = summary.loc[summary['metric'] == 'scaled_crps', 'NHITS'].values[0]
        
        print(f"Test MAE: {mae_val}")
        print(f"Test RMSE: {rmse_val}")
        print(f"Test Scaled CRPS: {crps_val}")
        
        # --- SUBSET EVALUATION (First 1000 samples) ---
        # To compare with notebook results which evaluated on a subset
        print("\n--- Subset Evaluation (First 1000 samples) ---")
        subset_eval_df = eval_df.iloc[:1000].copy()
        subset_evaluation_df = evaluate(
            subset_eval_df,
            metrics=metrics,
            models=['NHITS'],
            target_col='y',
            id_col='unique_id',
            time_col='ds',
            level=[80]
        )
        subset_summary = subset_evaluation_df.drop(columns=['unique_id']).groupby('metric').mean().reset_index()
        subset_mae = subset_summary.loc[subset_summary['metric'] == 'mae', 'NHITS'].values[0]
        subset_rmse = subset_summary.loc[subset_summary['metric'] == 'rmse', 'NHITS'].values[0]
        print(f"Subset MAE: {subset_mae}")
        print(f"Subset RMSE: {subset_rmse}")
        print("----------------------------------------------\n")
        
    except Exception as e:
        print(f"Error using utilsforecast: {e}. Falling back to manual calculation.", flush=True)
        # Fallback
        y_true = forecasts['y']
        y_pred = forecasts['NHITS-median'] if 'NHITS-median' in forecasts.columns else forecasts['NHITS']
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae_val = mean_absolute_error(y_true, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
        crps_val = None
        print(f"Test MAE: {mae_val}")
        print(f"Test RMSE: {rmse_val}")

    # Save Metrics
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
    
    if crps_val is not None:
        metrics_list.append({
            "name": "scaled_crps",
            "numberValue": float(crps_val),
            "format": "RAW"
        })
        
    metrics = {"metrics": metrics_list}
    
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_output_path}")

    # Plotting
    if plot_output_path:
        plot_loss(model_dir, plot_output_path)
        
    if prediction_plot_path:
        # Prepare metrics dictionary for display
        metrics_dict = {
            "MAE": mae_val,
            "RMSE": rmse_val
        }
        if crps_val is not None:
            metrics_dict["Scaled CRPS"] = crps_val
            
        plot_predictions(forecasts, prediction_plot_path, metrics_dict)

def plot_loss(model_dir, output_path):
    logs_dir = os.path.join(model_dir, "training_logs")
    if not os.path.exists(logs_dir):
        print("No training logs found.")
        # Write placeholder to ensure artifact exists
        with open(output_path, 'w') as f:
            f.write("<html><body><h1>No training logs found</h1></body></html>")
        return
        
    versions = [d for d in os.listdir(logs_dir) if d.startswith("version_")]
    if not versions:
        print("No version directory found in logs.")
        with open(output_path, 'w') as f:
            f.write("<html><body><h1>No version directory found in logs</h1></body></html>")
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
    # Use 'step' as x-axis if available, otherwise use index (which might be mixed steps/epochs)
    
    # Filter for epoch-level metrics if possible to avoid the high-frequency noise of step-level loss
    if 'epoch' in metrics_df.columns:
        epoch_df = metrics_df.groupby('epoch').mean().reset_index()
        if 'train_loss_epoch' in epoch_df.columns:
             plt.plot(epoch_df['epoch'], epoch_df['train_loss_epoch'], label='Train Loss')
        elif 'train_loss' in epoch_df.columns:
             plt.plot(epoch_df['epoch'], epoch_df['train_loss'], label='Train Loss')
             
        if 'valid_loss' in epoch_df.columns:
             plt.plot(epoch_df['epoch'], epoch_df['valid_loss'], label='Validation Loss')
    else:
        # Fallback to plotting everything (which looks noisy)
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

def plot_predictions(forecasts_df, output_path, metrics_dict=None):
    # Prepare data for plotting
    # Ensure we have actuals and predictions
    
    # Check for median or default
    y_pred_col = 'NHITS-median' if 'NHITS-median' in forecasts_df.columns else 'NHITS'
    
    # Fallback logic
    if y_pred_col not in forecasts_df.columns:
         candidates = [c for c in forecasts_df.columns if c.startswith('NHITS') and '-lo-' not in c and '-hi-' not in c]
         if candidates:
             y_pred_col = candidates[0]
    
    # Extract arrays
    actuals = forecasts_df['y'].values
    predictions = forecasts_df[y_pred_col].values
    ds = forecasts_df['ds'].values
    
    # Confidence Intervals
    lo_col = None
    hi_col = None
    if 'NHITS-lo-80.0' in forecasts_df.columns: lo_col = 'NHITS-lo-80.0'
    elif 'NHITS-lo-80' in forecasts_df.columns: lo_col = 'NHITS-lo-80'
    
    if 'NHITS-hi-80.0' in forecasts_df.columns: hi_col = 'NHITS-hi-80.0'
    elif 'NHITS-hi-80' in forecasts_df.columns: hi_col = 'NHITS-hi-80'

    # Create figure with GridSpec layout
    # Top: Time Series (Full Width)
    # Bottom: Density (Left), Residuals (Right)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    
    # --- Plot 1: Time Series (First 500 samples) ---
    ax1 = fig.add_subplot(gs[0, :])
    limit = 500
    
    # Slice for time series plot
    plot_ds = ds[:limit]
    plot_actuals = actuals[:limit]
    plot_preds = predictions[:limit]
    
    ax1.plot(plot_ds, plot_actuals, label='Actual', color='black', linewidth=1.5, alpha=1.0)
    ax1.plot(plot_ds, plot_preds, label='Predicted', color='blue', linewidth=1.5, alpha=0.7)
    
    # Add confidence intervals if available
    if lo_col and hi_col:
        plot_lo = forecasts_df[lo_col].values[:limit]
        plot_hi = forecasts_df[hi_col].values[:limit]
        ax1.fill_between(plot_ds, plot_lo, plot_hi, color='blue', alpha=0.2, label='80% Confidence Interval')
        
    ax1.set_title(f'Time Series Prediction (First {limit} samples)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Distribution of Actual vs Predicted ---
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Kernel Density Estimation
    try:
        # Remove NaNs for KDE
        clean_actuals = actuals[~np.isnan(actuals)]
        clean_preds = predictions[~np.isnan(predictions)]
        
        density_actual = gaussian_kde(clean_actuals)
        density_pred = gaussian_kde(clean_preds)
        
        min_val = min(clean_actuals.min(), clean_preds.min())
        max_val = max(clean_actuals.max(), clean_preds.max())
        padding = (max_val - min_val) * 0.1
        xs = np.linspace(min_val - padding, max_val + padding, 200)
        
        ax2.plot(xs, density_actual(xs), color='green', label='Actual')
        ax2.fill_between(xs, density_actual(xs), color='green', alpha=0.3)
        
        ax2.plot(xs, density_pred(xs), color='orange', label='Predicted')
        ax2.fill_between(xs, density_pred(xs), color='orange', alpha=0.3)
    except Exception as e:
        print(f"Could not plot KDE: {e}")
        ax2.hist(actuals, bins=30, density=True, alpha=0.5, color='green', label='Actual')
        ax2.hist(predictions, bins=30, density=True, alpha=0.5, color='orange', label='Predicted')
    
    ax2.set_title('Distribution of Actual vs Predicted Values')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Residuals ---
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = actuals - predictions
    
    # Remove NaNs
    residuals = residuals[~np.isnan(residuals)]
    
    # Histogram
    counts, bins, patches = ax3.hist(residuals, bins=30, density=False, color='purple', alpha=0.6, edgecolor='black')
    
    # KDE for residuals
    try:
        density_res = gaussian_kde(residuals)
        min_res = residuals.min()
        max_res = residuals.max()
        padding_res = (max_res - min_res) * 0.1
        xs_res = np.linspace(min_res - padding_res, max_res + padding_res, 200)
        
        curve = density_res(xs_res)
        bin_width = bins[1] - bins[0]
        scale_factor = len(residuals) * bin_width
        
        ax3.plot(xs_res, curve * scale_factor, color='purple', linewidth=2)
    except:
        pass
    
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax3.set_title('Distribution of Prediction Errors (Residuals)')
    ax3.set_xlabel('Error (Actual - Predicted)')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_plot_html(output_path, "Prediction Plot", metrics_dict)

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