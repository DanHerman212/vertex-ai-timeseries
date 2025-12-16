import argparse
import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from google.cloud import storage

def load_data_manual(input_path):
    print(f"Loading data from {input_path}...")
    with open(input_path) as f:
        data = f.read()
    lines = [line for line in data.split("\n") if line.strip()]
    header = lines[0].split(",")
    lines = lines[1:]
    
    mbt = np.zeros((len(lines),))
    # raw_data excludes the first column (date)
    raw_data = np.zeros((len(lines), len(header) - 1))
    
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(",")[1:]]
        mbt[i] = values[1] 
        raw_data[i, :] = values[:]
        
    return mbt, raw_data

def split_data_indices(n):
    num_train_samples = int(0.6 * n)
    num_val_samples = int(0.2 * n)
    num_test_samples = n - num_train_samples - num_val_samples
    
    test_start_idx = num_train_samples + num_val_samples
    return test_start_idx

def evaluate_gru(model_dir, raw_data, mbt, test_start_idx, sequence_length=150):
    print("Evaluating GRU Model...")
    # Check for SavedModel format (directory) or Keras file
    if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "saved_model.pb")):
        print(f"Loading SavedModel from {model_dir}...")
        # Use tf.saved_model.load for inference-only loading of SavedModels exported via model.export()
        # Note: model.export() creates a low-level SavedModel, not a Keras model.
        # We need to use the serving signature.
        loaded_model = tf.saved_model.load(model_dir)
        inference_func = loaded_model.signatures["serving_default"]
        
        print(f"Model inputs: {inference_func.structured_input_signature}")
        print(f"Model outputs: {inference_func.structured_outputs}")

        # We need to wrap this to behave like model.predict() for the evaluation loop below
        # or adjust the evaluation loop.
        # Let's create a simple wrapper function.
        def predict_wrapper(input_data):
            # input_data is numpy array (batch, seq_len, features)
            # Convert to tensor. Use tf.cast to handle float64 inputs safely.
            input_tensor = tf.cast(input_data, dtype=tf.float32)
            
            # Run inference
            # Check signature to determine if we need kwargs
            args, kwargs = inference_func.structured_input_signature
            
            try:
                if not args and kwargs:
                    # No positional args, but we have kwargs.
                    # Assuming single input for now, use the first key.
                    key = list(kwargs.keys())[0]
                    output = inference_func(**{key: input_tensor})
                else:
                    # Try positional
                    output = inference_func(input_tensor)
            except Exception as e:
                print(f"Inference call failed: {e}. Signature: {inference_func.structured_input_signature}")
                raise e
            
            # The output key is usually 'dense' or similar, or we take the first output
            # Let's inspect keys if needed, but usually it's the output layer name.
            # For a single output model, we can often just take the first value.
            return list(output.values())[0].numpy()
            
        model_predict = predict_wrapper
        
    else:
        # Fallback to looking for specific file if not a SavedModel dir
        model_path = os.path.join(model_dir, "gru_model.keras")
        if os.path.exists(model_path):
             print(f"Loading Keras model from {model_path}...")
             model = tf.keras.models.load_model(model_path)
             model_predict = model.predict
        else:
             # Try loading the directory itself (sometimes Keras saves as dir without saved_model.pb visible at top level in some versions)
             print(f"Attempting to load model from directory {model_dir}...")
             try:
                model = tf.keras.models.load_model(model_dir)
                model_predict = model.predict
             except:
                raise FileNotFoundError(f"Could not find valid model at {model_dir}")

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    # Load Scaler
    scaler = joblib.load(scaler_path)
    train_mean = scaler['mean']
    train_std = scaler['std']
    
    # Scale Data
    raw_data_scaled = (raw_data - train_mean) / train_std
    
    # Prepare Test Data (Sliding Window)
    # We need to start 'sequence_length' steps before the test_start_idx to predict the first test point
    start_idx = test_start_idx - sequence_length
    
    if start_idx < 0:
        raise ValueError("Not enough data history for the first test point.")

    # Create dataset
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=raw_data_scaled[start_idx:-1], # Input up to the last point
        targets=mbt[test_start_idx:],       # Targets starting from test_start_idx
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=128,
        shuffle=False
    )
    
    # Predict
    print("Generating predictions...")
    # Use the wrapper or the model.predict method
    if hasattr(model_predict, '__call__') and not isinstance(model_predict, tf.keras.Model):
         # It's our wrapper function, we need to iterate over the dataset manually or adjust wrapper
         # Since dataset is batched, let's iterate
         all_preds = []
         for batch in test_ds:
             inputs, _ = batch
             batch_preds = model_predict(inputs)
             all_preds.append(batch_preds)
         predictions_scaled = np.concatenate(all_preds, axis=0)
    else:
         predictions_scaled = model_predict(test_ds, verbose=1)
    
    # Model was trained on raw targets (mbt), so predictions are already in raw scale.
    # No inverse transform needed for the target variable.
    predictions = predictions_scaled
    
    # Get Actuals
    actuals = np.concatenate([y for x, y in test_ds], axis=0)
    
    mae = float(mean_absolute_error(actuals, predictions))
    print(f"GRU Test MAE: {mae}")
    return mae, actuals, predictions

def plot_loss(model_dir, output_path):
    history_path = os.path.join(model_dir, "history.json")
    if not os.path.exists(history_path):
        print(f"No history found at {history_path}, skipping plot.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    plt.figure(figsize=(10, 6))
    if 'loss' in history:
        plt.plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save as HTML for Vertex AI visualization
    # We can save as an image and embed it in HTML, or just save as image if the output type allows.
    # But Vertex AI 'HTML' artifact expects an HTML file.
    # Let's save the plot to a temporary image file, then embed it in HTML.
    
    # Actually, let's just save as a static image first, but since the artifact type is HTML,
    # we need to wrap it.
    
    import base64
    from io import BytesIO
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    html_content = f"""
    <html>
    <head>
        <title>Training Loss</title>
    </head>
    <body>
        <h1>Training Loss</h1>
        <img src="data:image/png;base64,{img_base64}" alt="Training Loss Plot">
    </body>
    </html>
    """
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Loss plot saved to {output_path}")

def plot_predictions(actuals, predictions, output_path):
    plt.figure(figsize=(12, 6))
    
    # Plot a subset if data is too large
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
    plt.ylabel('MBT')
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
    <head>
        <title>Prediction Plot</title>
    </head>
    <body>
        <h1>Actual vs Predicted</h1>
        <img src="data:image/png;base64,{img_base64}" alt="Prediction Plot">
    </body>
    </html>
    """
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Prediction plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--plot_output_path', type=str, required=False)
    parser.add_argument('--prediction_plot_path', type=str, required=False)
    args = parser.parse_args()

    # 1. Load Data
    mbt, raw_data = load_data_manual(args.input_csv)
    
    # 2. Determine Split
    n = len(raw_data)
    test_start_idx = split_data_indices(n)
    print(f"Total samples: {n}. Test starts at index: {test_start_idx}")
    
    # 3. Evaluate
    mae, actuals, predictions = evaluate_gru(args.model_dir, raw_data, mbt, test_start_idx)
    
    # 4. Plot Loss
    if args.plot_output_path:
        plot_loss(args.model_dir, args.plot_output_path)
        
    # 5. Plot Predictions
    if args.prediction_plot_path:
        plot_predictions(actuals, predictions, args.prediction_plot_path)
    
    # 6. Save Metrics for Vertex AI
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
    
    print(f"Metrics saved to {args.metrics_output_path}")
