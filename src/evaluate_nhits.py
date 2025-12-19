import argparse
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
        plot_predictions(forecasts, prediction_plot_path)

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

def plot_predictions(forecasts_df, output_path):
    plt.figure(figsize=(15, 6))
    
    # Plot a subset if too large
    limit = 500
    plot_df = forecasts_df.iloc[:limit]
    
    plt.plot(plot_df['ds'], plot_df['y'], label='Actual', color='black', alpha=0.7)
    
    # Check for median or default
    y_pred_col = 'NHITS-median' if 'NHITS-median' in plot_df.columns else 'NHITS'
    
    # If we renamed columns in eval_df, they might not be in forecasts_df passed here
    # Let's ensure we are plotting the right thing.
    if y_pred_col not in plot_df.columns:
         # Fallback to finding any column starting with NHITS that isn't lo/hi
         candidates = [c for c in plot_df.columns if c.startswith('NHITS') and '-lo-' not in c and '-hi-' not in c]
         if candidates:
             y_pred_col = candidates[0]
    
    plt.plot(plot_df['ds'], plot_df[y_pred_col], label='Predicted', color='blue', linewidth=2)
    
    # Plot Confidence Intervals if available
    # Note: We renamed columns for utilsforecast earlier, but forecasts_df might still have original names
    # or we might need to check for the renamed versions.
    
    lo_col = None
    hi_col = None
    
    if 'NHITS-lo-80.0' in plot_df.columns: lo_col = 'NHITS-lo-80.0'
    elif 'NHITS-lo-80' in plot_df.columns: lo_col = 'NHITS-lo-80'
    
    if 'NHITS-hi-80.0' in plot_df.columns: hi_col = 'NHITS-hi-80.0'
    elif 'NHITS-hi-80' in plot_df.columns: hi_col = 'NHITS-hi-80'

    if lo_col and hi_col:
        plt.fill_between(
            plot_df['ds'], 
            plot_df[lo_col], 
            plot_df[hi_col], 
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