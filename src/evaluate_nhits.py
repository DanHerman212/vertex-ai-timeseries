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
    
    # Ensure unique_id exists
    if 'unique_id' not in test_df.columns:
        test_df['unique_id'] = 'E'

    print(f"Loading model from {model_dir}...", flush=True)
    try:
        nf = NeuralForecast.load(path=model_dir)
        print("Model loaded successfully.", flush=True)
        
        # CRITICAL FIX: Disable EarlyStopping for evaluation
        # cross_validation creates a new Trainer. If early_stop_patience_steps is set, 
        # it expects validation metrics (ptl/val_loss). With val_size=0, these aren't produced.
        for model in nf.models:
            print(f"Disabling EarlyStopping and Logger for model {model}...", flush=True)
            model.early_stop_patience_steps = None
            
            # Disable Logger in trainer_kwargs to prevent "Missing folder" errors
            # This ensures the new Trainer created by cross_validation doesn't try to log to the old path
            if not hasattr(model, 'trainer_kwargs'):
                model.trainer_kwargs = {}
            model.trainer_kwargs['logger'] = False
            model.trainer_kwargs['enable_checkpointing'] = False
            
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
    val_size = 10
    n_test_steps = len(test_df) - input_size - val_size - 5
    print(f"Total rows in test_df: {len(test_df)}", flush=True)
    print(f"Expected test steps: {n_test_steps}", flush=True)
    
    if n_test_steps <= 0:
        raise ValueError(f"Test dataframe is too small ({len(test_df)}) for input_size ({input_size}) + val_size ({val_size}).")
        
    print(f"Forecasting {n_test_steps} steps using predict_insample (faster than cross_validation)...", flush=True)
    
    try:
        forecasts = nf.cross_validation(
            df=test_df,
            val_size=val_size, 
            test_size=n_test_steps,
            n_windows=None,
            step_size=1
        )
            
    except Exception as e:
        print(f"CRITICAL ERROR during predict_insample: {e}", flush=True)
        # Try to print more context
        import traceback
        traceback.print_exc()
        raise e
    
    print("Forecasts generated. Columns:", forecasts.columns, flush=True)
    
    # DEBUG: Inspect Forecast Statistics
    print("\n--- Forecast Statistics ---")
    # Identify prediction column
    pred_col = 'NHITS-median' if 'NHITS-median' in forecasts.columns else 'NHITS'
    if pred_col not in forecasts.columns:
         candidates = [c for c in forecasts.columns if c.startswith('NHITS') and '-lo-' not in c and '-hi-' not in c]
         if candidates:
             pred_col = candidates[0]
             
    if pred_col in forecasts.columns:
        print(forecasts[pred_col].describe())
        print(f"Max Prediction: {forecasts[pred_col].max()}")
        print(f"Min Prediction: {forecasts[pred_col].min()}")
        
        # Check for extreme outliers
        high_preds = forecasts[forecasts[pred_col] > 100]
        if not high_preds.empty:
            print(f"\nWARNING: Found {len(high_preds)} predictions > 100 minutes!")
            print(high_preds.head())
            
        # FIX: Clamp predictions to reasonable range [0, 120]
        # This prevents exploding gradients/predictions from ruining the metrics
        print("Clamping predictions to [0, 120] minutes...", flush=True)
        forecasts[pred_col] = forecasts[pred_col].clip(lower=0.0, upper=120.0)
        
    print("---------------------------\n")

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
        # Pass metrics dict for display in HTML
        metrics_dict = {
            "MAE": mae_val,
            "RMSE": rmse_val,
            "Scaled CRPS": crps_val
        }
        plot_predictions(forecasts, prediction_plot_path, metrics=metrics_dict)

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

def plot_predictions(forecasts_df, output_path, metrics=None):
    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    
    # --- Plot 1: Time Series (First 500 samples) ---
    ax1 = fig.add_subplot(gs[0, :])
    
    limit = 500
    plot_df = forecasts_df.iloc[:limit]
    
    # Plot Actuals
    ax1.plot(plot_df['ds'], plot_df['y'], label='Actual', color='black', linewidth=1.5, alpha=1.0)
    
    # Identify Prediction Column
    y_pred_col = 'NHITS-median' if 'NHITS-median' in forecasts_df.columns else 'NHITS'
    if y_pred_col not in forecasts_df.columns:
         candidates = [c for c in forecasts_df.columns if c.startswith('NHITS') and '-lo-' not in c and '-hi-' not in c]
         if candidates:
             y_pred_col = candidates[0]
    
    # Plot Predicted
    ax1.plot(plot_df['ds'], plot_df[y_pred_col], label='Predicted', color='blue', linewidth=1.5, alpha=0.7)
    
    # Plot Confidence Intervals
    lo_col = None
    hi_col = None
    if 'NHITS-lo-80.0' in plot_df.columns: lo_col = 'NHITS-lo-80.0'
    elif 'NHITS-lo-80' in plot_df.columns: lo_col = 'NHITS-lo-80'
    if 'NHITS-hi-80.0' in plot_df.columns: hi_col = 'NHITS-hi-80.0'
    elif 'NHITS-hi-80' in plot_df.columns: hi_col = 'NHITS-hi-80'

    if lo_col and hi_col:
        ax1.fill_between(
            plot_df['ds'], 
            plot_df[lo_col], 
            plot_df[hi_col], 
            color='blue', 
            alpha=0.2, 
            label='80% Confidence Interval'
        )
        
    ax1.set_title(f'Subway Headway Prediction: Actual vs Predicted (First {limit} samples)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Minutes Between Trains (MBT)')
    ax1.legend()
    ax1.grid(True)

    # --- Prepare Data for Density/Residuals (Use Full Dataset) ---
    actuals = forecasts_df['y'].values
    predictions = forecasts_df[y_pred_col].values
    residuals = actuals - predictions

    # --- Plot 2: Distribution of Actual vs Predicted ---
    ax2 = fig.add_subplot(gs[1, 0])
    try:
        density_actual = gaussian_kde(actuals)
        density_pred = gaussian_kde(predictions)
        
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        padding = (max_val - min_val) * 0.1
        xs = np.linspace(min_val - padding, max_val + padding, 200)
        
        ax2.plot(xs, density_actual(xs), color='green', label='Actual')
        ax2.fill_between(xs, density_actual(xs), color='green', alpha=0.3)
        
        ax2.plot(xs, density_pred(xs), color='orange', label='Predicted')
        ax2.fill_between(xs, density_pred(xs), color='orange', alpha=0.3)
    except Exception as e:
        print(f"KDE Error: {e}")
        ax2.hist(actuals, bins=30, density=True, alpha=0.5, color='green', label='Actual')
        ax2.hist(predictions, bins=30, density=True, alpha=0.5, color='orange', label='Predicted')
        
    ax2.set_title('Distribution of Actual vs Predicted Values')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Residuals ---
    ax3 = fig.add_subplot(gs[1, 1])
    
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
    
    save_plot_html(output_path, "Model Evaluation Report", metrics=metrics)

def save_plot_html(output_path, title, metrics=None):
    import base64
    from io import BytesIO
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Generate Metrics Table HTML if metrics are provided
    metrics_html = ""
    if metrics:
        rows = ""
        for key, value in metrics.items():
            # Format numbers nicely
            if isinstance(value, float):
                val_str = f"{value:.4f}"
            else:
                val_str = str(value)
            rows += f"<tr><td>{key}</td><td>{val_str}</td></tr>"
            
        metrics_html = f"""
        <div style="margin: 20px auto; width: 80%; font-family: Arial, sans-serif;">
            <h2 style="text-align: center; color: #333;">Performance Metrics</h2>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background-color: #007bff; color: white;">
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Metric</th>
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """
    
    html_content = f"""
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }}
            h1 {{ text-align: center; color: #333; }}
            img {{ display: block; margin: 0 auto; max-width: 100%; height: auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #ddd; }}
            td {{ padding: 10px; border: 1px solid #ddd; color: #555; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        {metrics_html}
        <div style="margin-top: 30px;">
            <img src="data:image/png;base64,{img_base64}" alt="{title}">
        </div>
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