import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow import keras

def plot_training_history(history, metrics=['mae']):
    """
    Plots the training and validation metrics from the history object.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    # Determine number of subplots
    num_plots = 1 + len(metrics)
    plt.figure(figsize=(6 * num_plots, 5))

    # Plot Loss
    plt.subplot(1, num_plots, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot additional metrics
    for i, metric in enumerate(metrics):
        if metric in history.history:
            plt.subplot(1, num_plots, i + 2)
            plt.plot(epochs, history.history[metric], label=f'Training {metric.upper()}')
            plt.plot(epochs, history.history[f'val_{metric}'], label=f'Validation {metric.upper()}')
            plt.title(f'Training and Validation {metric.upper()}')
            plt.xlabel('Epochs')
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model_path, test_dataset):
    """
    Loads the best model from disk and evaluates it on the test dataset using the Keras evaluate method.
    """
    print(f"Loading best model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    print("Evaluating model on test set...")
    results = model.evaluate(test_dataset, verbose=1)
    
    # Assuming metrics=['mae'] was used in compile
    # results will be [loss, mae]
    if isinstance(results, list) and len(results) > 1:
        print(f"Test MAE: {results[1]:.4f}")
        return results[1]
    else:
        print(f"Test Loss: {results:.4f}")
        return results

def plot_forecast_context(model_path, dataset, train_mean, train_std, num_samples=3):
    """
    Plots the input sequence history, the ground truth target, and the model prediction.
    Loads the best model from disk using model_path. Requires train_mean and train_std for inverse standard scaling.
    """
    model = keras.models.load_model(model_path)
    mbt_mean = train_mean[1]
    mbt_std = train_std[1]
    for x_batch, y_batch in dataset.take(1):
        preds = model.predict(x_batch, verbose=0)
        for i in range(num_samples):
            input_seq_scaled = x_batch[i, :, 1]
            input_seq_real = (input_seq_scaled * mbt_std) + mbt_mean
            true_val = float(y_batch[i][0])
            pred_val = float(preds[i][0])
            abs_error = abs(true_val - pred_val)
            plt.figure(figsize=(12, 5))
            time_steps = range(len(input_seq_real))
            plt.plot(time_steps, input_seq_real, label='History (Input)', color='#1f77b4')
            plt.plot(len(input_seq_real), true_val, 'go', markersize=10, 
                     label=f'True Future: {true_val:.2f} min')
            plt.plot(len(input_seq_real), pred_val, 'rx', markersize=10, markeredgewidth=2, 
                     label=f'Prediction: {pred_val:.2f} min')
            plt.text(len(input_seq_real) + 2, true_val, f'{true_val:.2f}', color='green', va='center', fontweight='bold')
            plt.text(len(input_seq_real) + 2, pred_val, f'{pred_val:.2f}', color='red', va='center', fontweight='bold')
            plt.title(f'Sample {i+1}: History vs Prediction (Abs Error: {abs_error:.2f} min)')
            plt.xlabel('Time Steps')
            plt.ylabel('Minutes Between Trains')
            plt.xlim(0, len(input_seq_real) + 15)
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.show()
        break

def create_results_df(model_path, dataset, mean, std):
    """
    Loads a model from disk, generates predictions, and returns a DataFrame with actual, predicted, error, and abs error columns in original units.
    Args:
        model_path (str): Path to the saved Keras model.
        dataset (tf.data.Dataset): Dataset to predict on.
        mean (float): Mean used for inverse transform.
        std (float): Std used for inverse transform.
    Returns:
        pd.DataFrame: DataFrame with columns ['Actual', 'Predicted', 'Error', 'Abs Error'] in original units.
    """
    model = keras.models.load_model(model_path)
    all_actuals = []
    all_preds = []
    for x_batch, y_batch in dataset:
        batch_preds = model.predict(x_batch, verbose=0)
        all_actuals.extend(y_batch.numpy().flatten())
        all_preds.extend(batch_preds.flatten())
    results_df = pd.DataFrame({
        'Actual': all_actuals,
        'Predicted': all_preds
    })
    # Inverse transform
    results_df['Actual'] = results_df['Actual'] * std + mean
    results_df['Predicted'] = results_df['Predicted'] * std + mean
    results_df['Error'] = results_df['Actual'] - results_df['Predicted']
    results_df['Abs Error'] = results_df['Error'].abs()
    return results_df

def plot_predictions_and_errors(results_df):
    """
    Plots two figures:
    1. Distribution of predictions vs actuals (KDE plot).
    2. Histogram of errors with kernel density line.
    Args:
        results_df (pd.DataFrame): DataFrame with 'Actual', 'Predicted', and 'Error' columns.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Plot 1: Distribution of predictions vs actuals
    plt.figure(figsize=(12, 6))
    sns.kdeplot(results_df['Actual'], label='Actual', fill=True, alpha=0.3, color='green')
    sns.kdeplot(results_df['Predicted'], label='Predicted', fill=True, alpha=0.3, color='orange')
    plt.title('Distribution of Actual vs Predicted Values')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot 2: Histogram of errors with kernel density
    plt.figure(figsize=(12, 6))
    sns.histplot(results_df['Error'], bins=50, kde=True, color='purple', alpha=0.7)
    plt.title('Distribution of Prediction Errors (Residuals)')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Count')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.show()

