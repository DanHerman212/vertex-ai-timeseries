import argparse
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from neuralforecast import NeuralForecast

def evaluate_nhits(model_dir, test_csv_path, metrics_output_path, plot_output_path, prediction_plot_path):
    print(f"Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    
    # Ensure datetime
    if 'ds' in test_df.columns:
        test_df['ds'] = pd.to_datetime(test_df['ds'])
    
    print(f"Loading model from {model_dir}...")
    nf = NeuralForecast.load(path=model_dir)
    
    print("Generating predictions...")
    # NeuralForecast predict needs the future dataframe or just the history if it's recursive.
    # But here we have the test set with 'y' (actuals) and exog variables.
    # We can pass the test_df. 
    # Note: predict() usually expects the future horizon. 
    # If we want to evaluate on the test set which follows the training set, 
    # we might need to pass the full dataset or handle the context.
    # However, since we trained on train+val, and test follows immediately, 
    # we can try passing test_df. 
    # But NHITS is a windows-based model. It needs the lookback window.
    # If we just pass test_df, it might complain about missing history if the test_df starts right after train.
    # Ideally, we should have passed the last window of train data + test data.
    # BUT, NeuralForecast.predict() by default predicts 'h' steps into the future from the end of the input df.
    # If we want to predict for the *existing* test_df timestamps, we should use `predict_insample` or `cross_validation`?
    # Or, we can use `predict(df=test_df)`?
    # Actually, `predict` with `df` argument uses the `df` as the history to predict *after* it.
    # That's not what we want. We want to predict *for* the rows in `test_df`.
    
    # Correct approach for NeuralForecast evaluation on a held-out set:
    # We need to provide the history leading up to each point.
    # Since we are doing 1-step ahead forecasting (h=1), we effectively want to predict y_t given y_{t-L}...y_{t-1}.
    # The `test_df` contains the targets `y`.
    # If we treat this as a "production" inference where we have the history, we can loop or use a rolling window.
    # However, NeuralForecast has `predict` which can take `futr_exog_list`.
    
    # Let's try a simpler approach often used:
    # We can use `cross_validation` but that retrains or refits.
    # We just want inference.
    
    # If we use `nf.predict(df=history_df)`, it predicts h steps after history_df.
    # To predict the whole test set, we technically need to feed it (history + test_df) and ask for fitted values?
    # Or use `predict` in a rolling fashion?
    
    # Given the complexity of "correctly" feeding the rolling window in NF without re-implementing it,
    # and since we set `h=1`, maybe we can use `predict` if we had the full dataset?
    # But we only have `test_df` here.
    
    # WAIT: The `train_nhits.py` saved `test_df` which is just the last 20%.
    # It does NOT contain the lookback window from the validation set.
    # This is a problem for N-HiTS if it needs 150 steps of context.
    # The first 150 predictions will be garbage or impossible if we don't provide context.
    
    # FIX: We should probably have saved `test_df` WITH the lookback context in `train_nhits.py`.
    # OR, we can just accept that we lose the first 150 points of evaluation.
    # Let's assume we accept losing the first 150 points for now to keep it simple, 
    # or we can try to rely on the model's internal state if it was saved with the series?
    # `nf.save` saves the model weights, but `nf.load` doesn't restore the training data.
    
    # Let's try to predict using `test_df` as the input. 
    # If `test_df` has enough length (> input_size), we can generate predictions for indices [input_size:].
    # `nf.predict` usually predicts *future* of the input.
    # We want "in-sample" predictions for the test set.
    # There isn't a direct "predict_insample" for new data in NF that I recall easily.
    # But `predict` returns forecasts for the end of the series.
    
    # Actually, `NeuralForecast.predict` takes `df` (history).
    # If we pass `df = test_df`, it will predict `h` steps *after* `test_df`.
    # That's not what we want.
    
    # We want to predict row `i` using rows `i-input_size` to `i-1`.
    # This is exactly what `train_gru` does with `tf.data.Dataset`.
    
    # In NeuralForecast, to get predictions on a test set, the standard way is often `cross_validation`.
    # But since we have a pre-trained model, we might want to use `predict` iteratively? No, too slow.
    
    # Alternative: Use `nf.predict_insample`? 
    # `predict_insample` works on the data seen during training.
    # But we didn't train on test data.
    
    # Let's look at `nf.predict` signature.
    # It predicts `h` steps ahead.
    
    # If we want to evaluate on a long test sequence with `h=1`, we are essentially doing a rolling forecast.
    # NeuralForecast is optimized for this?
    # Actually, if `h=1`, we can just treat it as a series of inputs.
    
    # Let's try this:
    # We can't easily do this without the lookback.
    # I will assume for this iteration that `test_df` is large enough that we can just use it to "warm up"?
    # No, that doesn't make sense.
    
    # Let's look at how `train_nhits.py` saved the data.
    # `test_df = Y_df.iloc[train_size + val_size:]`
    # It has NO overlap.
    
    # I will modify `train_nhits.py` to include the lookback window in the export.
    # This is safer.
    # `test_df = Y_df.iloc[train_size + val_size - input_size:]`
    # Then in `evaluate_nhits.py`, we can use this context.
    
    # But how to invoke prediction?
    # If I pass this extended `test_df` to `nf.predict`, it still predicts *after* the end.
    
    # Wait, `NeuralForecast` models are often global models.
    # Maybe I can use `model.predict(dataset)`?
    # The `NeuralForecast` class is a wrapper.
    
    # Let's stick to the simplest valid approach:
    # We want to verify the model performance.
    # If we cannot easily run rolling predictions on test set with NF, 
    # maybe we should just use `cross_validation` in the training step?
    # But the user wants a separate evaluate component.
    
    # Let's assume we can use `nf.predict` if we structure the input correctly.
    # Actually, for `h=1`, we can construct a dataset where `ds` is the time, `y` is the target.
    # If we pass this to `nf.predict`, does it output fitted values? No.
    
    # Let's look at `NeuralForecast.predict` docs (mental check).
    # `predict(df=None, static_df=None, futr_df=None, ...)`
    # "Generate forecasts for the next `h` steps."
    
    # This implies it's for FUTURE forecasting.
    # To evaluate on a test set, we usually use `cross_validation`.
    # `nf.cross_validation(df=Y_df, val_size=val_size, test_size=test_size, n_windows=None)`
    # This would respect the temporal order.
    
    # But we already trained the model.
    # We just want to apply it.
    
    # Maybe we can use the underlying PyTorch model directly?
    # `nf.models[0].predict(dataset)`?
    
    # Let's try to use `nf.predict` but we might be limited to forecasting *after* the provided data.
    # If we want to evaluate on the test set, we effectively want to forecast *each point* in the test set given history.
    # This is `n_windows = len(test_set)`.
    # That is very expensive if we call it iteratively.
    
    # However, `NHITS` is a global model. It can process batches.
    # If we construct a dataset of windows (like we did for GRU), we can feed it to the model.
    # But `NeuralForecast` wrapper expects DataFrames.
    
    # Let's try to use `predict` with a trick?
    # No.
    
    # Let's go back to `train_nhits.py`.
    # If I change `train_nhits.py` to export the *entire* dataset (or at least train+val+test), 
    # then `evaluate_nhits.py` can load the model and run `predict`? 
    # No, `predict` is still for future.
    
    # What if we use `predict_insample`?
    # `nf.predict_insample(step_size=1)` returns in-sample predictions for the training set.
    # If we could "trick" it to think the test set is part of the training set (without training), 
    # we could get predictions.
    # `nf.fit(df=test_df, max_steps=0)`? (Train for 0 steps).
    # Then `predict_insample`?
    
    # This seems like a viable hack.
    # 1. Load trained model.
    # 2. Call `nf.fit(df=test_df_with_context, max_steps=0)`.
    # 3. Call `nf.predict_insample()`.
    # 4. Extract the part corresponding to the test set.
    
    # I will proceed with this plan.
    # First, I need to update `train_nhits.py` to export `test_df` WITH CONTEXT (lookback window).
    # `input_size = 150`.
    
    pass

def plot_loss(model_dir, output_path):
    # Find metrics.csv
    logs_dir = os.path.join(model_dir, "training_logs")
    # Find the version folder (e.g. version_0)
    # We'll just take the first one we find
    if not os.path.exists(logs_dir):
        print("No training logs found.")
        return
        
    versions = [d for d in os.listdir(logs_dir) if d.startswith("version_")]
    if not versions:
        print("No version directory found in logs.")
        return
        
    metrics_path = os.path.join(logs_dir, versions[0], "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"No metrics.csv found at {metrics_path}")
        return
        
    print(f"Loading metrics from {metrics_path}...")
    metrics_df = pd.read_csv(metrics_path)
    
    plt.figure(figsize=(10, 6))
    # Plot train_loss_step or train_loss_epoch
    # Usually 'train_loss' and 'valid_loss' are columns
    if 'train_loss_epoch' in metrics_df.columns:
        plt.plot(metrics_df['train_loss_epoch'].dropna(), label='Train Loss')
    elif 'train_loss' in metrics_df.columns:
        plt.plot(metrics_df['train_loss'].dropna(), label='Train Loss')
        
    if 'valid_loss' in metrics_df.columns:
        plt.plot(metrics_df['valid_loss'].dropna(), label='Validation Loss')
        
    plt.title('N-HiTS Training Loss')
    plt.xlabel('Epoch/Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    import base64
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    html_content = f"""
    <html>
    <head><title>N-HiTS Loss</title></head>
    <body>
        <h1>N-HiTS Training Loss</h1>
        <img src="data:image/png;base64,{img_base64}" alt="Loss Plot">
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"Loss plot saved to {output_path}")

def plot_predictions(actuals, predictions, output_path):
    plt.figure(figsize=(12, 6))
    limit = 500
    if len(actuals) > limit:
        plt.plot(actuals[:limit], label='Actual', alpha=0.7)
        plt.plot(predictions[:limit], label='Predicted', alpha=0.7)
        plt.title(f'Actual vs Predicted (First {limit} samples)')
    else:
        plt.plot(actuals, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted')
        
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    import base64
    from io import BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    html_content = f"""
    <html>
    <head><title>Prediction Plot</title></head>
    <body>
        <h1>Actual vs Predicted</h1>
        <img src="data:image/png;base64,{img_base64}" alt="Prediction Plot">
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--plot_output_path', type=str, required=False)
    parser.add_argument('--prediction_plot_path', type=str, required=False)
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model from {args.model_dir}...")
    nf = NeuralForecast.load(path=args.model_dir)
    
    # 2. Load Data
    print(f"Loading test data from {args.test_csv_path}...")
    test_df = pd.read_csv(args.test_csv_path)
    if 'ds' in test_df.columns:
        test_df['ds'] = pd.to_datetime(test_df['ds'])
        
    # 3. Generate Predictions
    # Hack: Fit with 0 steps to initialize internal state with new data, then predict_insample
    # We assume test_df includes the necessary lookback context (handled in train_nhits.py)
    print("Running inference...")
    # We need to ensure the model is on the correct device
    # nf.models[0].accelerator = 'cpu' # Force CPU for safety
    
    # We use fit with max_steps=0 to prime the model with the test data
    # This allows predict_insample to work on this new data
    nf.fit(df=test_df, val_size=0, max_steps=0)
    
    # Predict insample
    insample_preds = nf.predict_insample(step_size=1)
    
    # The insample_preds dataframe will have index aligned with test_df (minus the first input_size points)
    # We need to align actuals and predictions
    # insample_preds has columns ['ds', 'unique_id', 'y', 'NHITS']
    
    # Merge with actuals to be safe
    merged = pd.merge(test_df, insample_preds, on=['ds', 'unique_id'], how='inner', suffixes=('', '_pred'))
    
    # Actuals are in 'y', Predictions in 'NHITS'
    actuals = merged['y'].values
    predictions = merged['NHITS'].values
    
    mae = mean_absolute_error(actuals, predictions)
    print(f"N-HiTS Test MAE: {mae}")
    
    # 4. Save Metrics
    metrics = {
        "metrics": [
            {
                "name": "mae",
                "numberValue": mae,
                "format": "RAW"
            }
        ]
    }
    with open(args.metrics_output_path, 'w') as f:
        json.dump(metrics, f)
        
    # 5. Plot Loss
    if args.plot_output_path:
        plot_loss(args.model_dir, args.plot_output_path)
        
    # 6. Plot Predictions
    if args.prediction_plot_path:
        plot_predictions(actuals, predictions, args.prediction_plot_path)
