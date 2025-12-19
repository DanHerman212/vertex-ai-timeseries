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
        
    # SMOKE TEST: Try to predict just 2 steps first to verify everything works
    # Removed smoke test as it was failing for the same reason (val_size=0).
    # We will fix the main call directly.

    print(f"Forecasting {n_test_steps} steps using predict_insample (faster than cross_validation)...", flush=True)
    
    try:
        # OPTIMIZATION: Use predict_insample instead of cross_validation
        # predict_insample generates 1-step ahead forecasts for the provided dataframe
        # using the ground truth 'y' as inputs. This is equivalent to rolling forecast with h=1
        # but runs as a batched inference operation (seconds) instead of a loop (hours).
        # Note: predict_insample does not take 'df' argument in newer versions, it uses the internal dataset.
        # However, since we loaded the model from disk, it doesn't have the dataset.
        # We must use predict_insample(step_size=1) if the model was just trained, 
        # BUT since we are loading a fresh model, we should use `predict` with `futr_df` 
        # OR we can use `cross_validation` with `step_size=1` (slow)
        # OR we can trick it.
        
        # Actually, predict_insample is for the training set.
        # For a NEW test set, we want to generate 1-step ahead forecasts given the history.
        # The correct method for this in NeuralForecast is actually `cross_validation` with `step_size=1` (which is slow)
        # OR `predict` if we just want the horizon.
        
        # Wait, if we want 1-step ahead for the WHOLE test set, we can treat the test set as a new "training" set
        # for the purpose of generating insample predictions (without training).
        # We can use `nf.predict_insample` but we need to pass the dataframe?
        # Checking docs: predict_insample(step_size=1) uses the data stored in the object.
        
        # Workaround: We can use `nf.predict` iteratively? No, that's slow.
        
        # Let's try to use `nf.predict` but passing the history?
        # No, `predict` is for future.
        
        # The fast way to get 1-step ahead forecasts for a new dataset is to use `cross_validation` 
        # BUT with a trick or a different method?
        # Actually, `predict_insample` DOES take `df` in some versions, but apparently not this one (1.6.4).
        # Let's check the installed version. It is 1.6.4 in requirements.nhits.txt.
        # In 1.6.4, predict_insample might not exist or work this way.
        
        # Alternative Fast Method:
        # We can use the model's `predict` method directly if we prepare the batches.
        # But that's complex.
        
        # Let's revert to `cross_validation` but optimize it?
        # No, `cross_validation` is inherently slow because it refits or re-predicts.
        
        # WAIT! If we just want to evaluate 1-step ahead accuracy on the test set,
        # we can treat the test set as "new data" and ask for fitted values?
        # No.
        
        # Let's look at the error: `TypeError: NeuralForecast.predict_insample() got an unexpected keyword argument 'df'`
        # This means `predict_insample` exists but doesn't take `df`. It only predicts on the data it was trained on.
        
        # Since we loaded the model from disk, it has NO data.
        # We need to "feed" the data to it first?
        # nf.dataset = ...?
        
        # Better approach for 1.6.4:
        # Use `nf.predict` with `futr_df`? No.
        
        # Let's try `nf.predict_insample` AFTER calling `nf.fit`? 
        # But we don't want to retrain.
        
        # There is a method `predict_rolling` in newer versions?
        # Let's check if we can use `cross_validation` but with a larger step size?
        # The user specifically wants 1-step ahead.
        
        # Let's try to use the internal model's forward pass?
        
        # Actually, `predict_insample` in 1.6.4 might just be `predict_insample(step_size=1)`.
        # But how do we give it the new data?
        # We can't.
        
        # So we MUST use `cross_validation`.
        # BUT, we can speed it up by NOT refitting. `refit=False` is default.
        # The slowness comes from the loop.
        
        # Is there a `predict_rolling`?
        # Let's try to use `nf.predict` but we need to update the state.
        
        # Let's go back to `cross_validation` but maybe we can optimize the `input_size`?
        # No.
        
        # Wait, `predict_insample` was added recently.
        # If we can't use it, we are stuck with `cross_validation`.
        
        # However, we can try to use `nf.models[0].predict` directly?
        
        # Let's try to use `cross_validation` again but ensure we are not doing something wrong.
        # The error before was about timezone.
        # Maybe `cross_validation` IS the right way, and it was just failing on timezone?
        # But the user said it was "taking very long".
        
        # Let's try to use `predict` with a loop but in batches?
        # No.
        
        # Let's try to use `nf.predict` on the whole test set?
        # `nf.predict(df=test_df)`?
        # If we pass `df` to `predict`, it uses it as history and predicts `h` steps into the future.
        # That's not what we want. We want 1-step ahead for EACH point in `df`.
        
        # Let's try to use `cross_validation` but with `step_size=1`.
        # If it's too slow, we might need to upgrade NeuralForecast?
        # Or we can use a custom loop that is faster.
        
        # Let's try to use `cross_validation` again. 
        # The timezone fix might have solved the ERROR, but not the SPEED.
        # But maybe the speed issue was exaggerated or due to something else?
        # 21,000 steps with a deep learning model in a loop in Python IS slow.
        
        # Let's try to use `predict_insample` by hacking the object?
        # nf.dataset = ...
        
        # Let's try to use `cross_validation` but verify if it works now with the timezone fix.
        # If it works, at least we have a working pipeline.
        # We can optimize speed later or reduce the test set size.
        
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