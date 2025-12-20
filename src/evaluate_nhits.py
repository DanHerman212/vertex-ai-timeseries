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
from matplotlib.dates import DateFormatter
from neuralforecast import NeuralForecast
from utilsforecast.losses import mae, rmse
from utilsforecast.evaluation import evaluate
import warnings
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

def get_plot_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_cv_results(cv_df):
    # Plot the last 200 points
    plot_df = cv_df.tail(200)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(plot_df['ds'], plot_df['y'], label='Actual MBT', color='black', alpha=0.7)
    
    # Identify Model Column (renamed to NHITS)
    if 'NHITS' in plot_df.columns:
        ax.plot(plot_df['ds'], plot_df['NHITS'], label='Predicted Median', color='blue', linewidth=2)
    
    # Plot Confidence Interval
    # Check for available interval columns
    # MQLoss with quantiles 0.1, 0.9 produces 80% interval
    lo_col = None
    hi_col = None
    
    # Common naming patterns from NeuralForecast
    if 'NHITS-lo-80.0' in plot_df.columns and 'NHITS-hi-80.0' in plot_df.columns:
        lo_col = 'NHITS-lo-80.0'
        hi_col = 'NHITS-hi-80.0'
        label = '80% Confidence Interval'
    elif 'NHITS-lo-90' in plot_df.columns and 'NHITS-hi-90' in plot_df.columns:
        lo_col = 'NHITS-lo-90'
        hi_col = 'NHITS-hi-90'
        label = '90% Confidence Interval'
        
    if lo_col and hi_col:
        ax.fill_between(plot_df['ds'], plot_df[lo_col], plot_df[hi_col], color='blue', alpha=0.2, label=label)

    ax.set_title('NHITS Forecast vs Actuals (Cross Validation)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Minutes Between Trains (MBT)')
    
    # Format x-axis
    date_form = DateFormatter("%Y-%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_form)
    
    ax.legend()
    ax.grid(True, alpha=0.3)

    img_base64 = get_plot_base64(fig)
    plt.close(fig)
    return img_base64

def generate_html_report(output_path, metrics_dict, pred_plot_b64):
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
        <title>NHITS Model Evaluation (CV)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border: 1px solid #ddd; }}
            th, td {{ text-align: left; padding: 8px; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            h1, h3 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>NHITS Model Evaluation Report (Cross Validation)</h1>
        <p>Method: Rolling Forecast via NeuralForecast.cross_validation</p>
        {metrics_html}
        {pred_html}
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"HTML report saved to {output_path}", flush=True)

def evaluate_nhits(model_dir, df_csv_path, metrics_output_path, html_output_path):
    print(f"Starting evaluation script (Cross Validation mode)...", flush=True)
    
    # 1. Load Data
    print(f"Loading data from {df_csv_path}...", flush=True)
    df = pd.read_csv(df_csv_path)
    
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
    if 'unique_id' not in df.columns:
        df['unique_id'] = 'E'
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    print(f"Data shape: {df.shape}", flush=True)

    # 2. Load Model
    print(f"Loading model from {model_dir}...", flush=True)
    nf = NeuralForecast.load(path=model_dir)
    
    # WORKAROUND: Fix "Missing folder" error when loading model in new environment
    # The model tries to log to the original training directory which doesn't exist.
    # We disable logging and checkpointing for inference.
    try:
        for i, model in enumerate(nf.models):
            print(f"Patching model {i}...", flush=True)
            if hasattr(model, 'trainer_kwargs'):
                print(f"  Original trainer_kwargs keys: {model.trainer_kwargs.keys()}", flush=True)
                model.trainer_kwargs['logger'] = False
                model.trainer_kwargs['enable_checkpointing'] = False
                
                # Also ensure accelerator matches current environment
                import torch
                model.trainer_kwargs['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'
                
                # Remove callbacks that might fail or are unnecessary for inference
                if 'callbacks' in model.trainer_kwargs:
                    model.trainer_kwargs['callbacks'] = []
                
                print(f"  Patched trainer_kwargs: logger={model.trainer_kwargs.get('logger')}, accelerator={model.trainer_kwargs.get('accelerator')}", flush=True)
                
            if hasattr(model, '_logger'): model._logger = None
            if hasattr(model, '_trainer'): model._trainer = None
        print("Applied logger/trainer workaround for inference.", flush=True)
    except Exception as e:
        print(f"Warning: Could not apply logger workaround: {e}", flush=True)

    print("Model loaded successfully.", flush=True)

    # 3. Run Cross Validation
    # Evaluate on last 20%
    total_rows = len(df)
    test_size = int(total_rows * 0.2)
    print(f"Running Cross Validation on last {test_size} steps...", flush=True)
    
    cv_df = nf.cross_validation(
        df=df,
        test_size=test_size,
        n_windows=None,
        step_size=1,
        refit=False,
        verbose=False
    )
    print("Cross Validation completed.", flush=True)
    
    # 4. Post-process
    # Rename median column to model name for easier handling
    if 'NHITS-median' in cv_df.columns:
        cv_df = cv_df.rename(columns={'NHITS-median': 'NHITS'})
    
    # 5. Calculate Metrics
    print("Calculating metrics...", flush=True)
    metrics = [mae, rmse]
    evaluation = evaluate(
        cv_df.drop(columns=['cutoff'], errors='ignore'),
        metrics=metrics,
        models=['NHITS']
    )
    
    # Extract scalar values
    mae_val = evaluation.loc[evaluation['metric'] == 'mae', 'NHITS'].values[0]
    rmse_val = evaluation.loc[evaluation['metric'] == 'rmse', 'NHITS'].values[0]
    
    print(f"MAE: {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    
    # Save Metrics JSON
    metrics_list = [
        {"name": "mae", "numberValue": float(mae_val), "format": "RAW"},
        {"name": "rmse", "numberValue": float(rmse_val), "format": "RAW"}
    ]
    metrics_json = {"metrics": metrics_list}
    
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics_json, f)
    print(f"Metrics saved to {metrics_output_path}")

    # 6. Generate Report
    print("Generating plots...", flush=True)
    pred_plot_b64 = plot_cv_results(cv_df)
    
    metrics_dict = {"MAE": mae_val, "RMSE": rmse_val}
    generate_html_report(html_output_path, metrics_dict, pred_plot_b64)
    print("Evaluation finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_csv_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--html_output_path', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=False) # Kept for compatibility
    
    args = parser.parse_args()
    
    evaluate_nhits(
        args.model_dir, 
        args.df_csv_path, 
        args.metrics_output_path, 
        args.html_output_path
    )
