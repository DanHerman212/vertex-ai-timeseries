import argparse
import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Set backend to Agg for headless environments
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error
from google.cloud import storage

def evaluate_gru(model_dir, test_ds):
    print("Evaluating GRU Model...")
    
    model_path = os.path.join(model_dir, "gru_model.keras")
    if os.path.exists(model_path):
            print(f"Loading Keras model from {model_path}...")
            model = tf.keras.models.load_model(model_path)
    else:
            raise FileNotFoundError(f"Could not find valid model at {model_path}")

    # Predict
    print("Generating predictions...")
    predictions = model.predict(test_ds, verbose=1)
    
    # Model was trained on raw targets (mbt), so predictions are already in raw scale.
    # No inverse transform needed for the target variable.
    
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

    # Create figure with two subplots
    plt.figure(figsize=(16, 6))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['loss']) + 1)
    if 'loss' in history:
        plt.plot(epochs, history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: MAE
    plt.subplot(1, 2, 2)
    if 'mae' in history:
        plt.plot(epochs, history['mae'], label='Training MAE')
    if 'val_mae' in history:
        plt.plot(epochs, history['val_mae'], label='Validation MAE')
        
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
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
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Loss plot saved to {output_path}")

def plot_residuals_distribution(actuals, predictions, output_path):
    # Ensure 1D arrays for KDE
    actuals = actuals.flatten()
    predictions = predictions.flatten()

    plt.figure(figsize=(16, 6))

    # Plot 1: Distribution of Actual vs Predicted
    plt.subplot(1, 2, 1)
    
    # Kernel Density Estimation
    density_actual = gaussian_kde(actuals)
    density_pred = gaussian_kde(predictions)
    
    # Determine range for evaluation
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    padding = (max_val - min_val) * 0.1
    xs = np.linspace(min_val - padding, max_val + padding, 200)
    
    plt.plot(xs, density_actual(xs), color='green', label='Actual')
    plt.fill_between(xs, density_actual(xs), color='green', alpha=0.3)
    
    plt.plot(xs, density_pred(xs), color='orange', label='Predicted')
    plt.fill_between(xs, density_pred(xs), color='orange', alpha=0.3)
    
    plt.title('Distribution of Actual vs Predicted Values')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = actuals - predictions.flatten() # Ensure 1D
    plt.subplot(1, 2, 2)
    
    # Histogram
    # Use density=False to get counts, but we need to scale KDE to match
    counts, bins, patches = plt.hist(residuals, bins=30, density=False, color='purple', alpha=0.6, edgecolor='black')
    
    # KDE for residuals
    density_res = gaussian_kde(residuals)
    
    min_res = residuals.min()
    max_res = residuals.max()
    padding_res = (max_res - min_res) * 0.1
    xs_res = np.linspace(min_res - padding_res, max_res + padding_res, 200)
    
    curve = density_res(xs_res)
    # Scale curve to match histogram counts
    # Area of histogram = sum(counts) * bin_width
    # Area of PDF = 1
    # So scale factor = len(residuals) * bin_width
    bin_width = bins[1] - bins[0]
    scale_factor = len(residuals) * bin_width
    
    plt.plot(xs_res, curve * scale_factor, color='purple', linewidth=2)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    plt.title('Distribution of Prediction Errors (Residuals)')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    import base64
    from io import BytesIO
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    html_content = f"""
    <html>
    <head>
        <title>Residuals and Distribution</title>
    </head>
    <body>
        <h1>Model Evaluation Plots</h1>
        <img src="data:image/png;base64,{img_base64}" alt="Residuals and Distribution Plot">
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Residuals and distribution plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--plot_output_path', type=str, required=False)
    parser.add_argument('--prediction_plot_path', type=str, required=False)
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading test dataset from {args.test_dataset_path}...")
    test_ds = tf.data.Dataset.load(args.test_dataset_path)
    
    # 2. Evaluate
    mae, actuals, predictions = evaluate_gru(args.model_dir, test_ds)
    
    # 3. Plot Loss
    if args.plot_output_path:
        plot_loss(args.model_dir, args.plot_output_path)
        
    # 4. Plot Residuals and Distribution
    if args.prediction_plot_path:
        plot_residuals_distribution(actuals, predictions, args.prediction_plot_path)
    
    # 5. Save Metrics for Vertex AI
    metrics = {
        "metrics": [
            {
                "name": "mae",
                "numberValue": mae,
                "format": "RAW"
            }
        ]
    }
    
    os.makedirs(os.path.dirname(args.metrics_output_path), exist_ok=True)
    with open(args.metrics_output_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Metrics saved to {args.metrics_output_path}")
