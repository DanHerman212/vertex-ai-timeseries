import pandas as pd
import numpy as np
import argparse
import os

def preprocess_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    
    # Load Data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found. Please ensure the BigQuery extraction step has completed successfully.")
        
    df = pd.read_csv(input_path)
    
    # Validate expected schema from BigQuery
    required_columns = ['arrival_date', 'duration', 'mbt']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}. Expected columns: {required_columns}")

    # Ensure datetime
    if 'arrival_date' in df.columns:
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        # Rename for NeuralForecast compatibility (Universal standard here)
        df = df.rename(columns={'arrival_date': 'ds', 'mbt': 'y'})
    elif 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
    
    # Filter years if needed (from TF notebook)
    df['year'] = df['ds'].dt.year
    df = df[df['year'].isin([2024, 2025])].copy()
    df = df.drop(columns=['year'])

    # --- Feature Engineering (Universal) ---
    print("Generating features...")
    
    # 1. Rolling Features
    # Calculate rolling std with window 50
    df['rolling_std_50'] = df['y'].rolling(window=50).std().bfill()
    # Add Rolling Max to capture recent spikes (delays)
    df['rolling_max_10'] = df['y'].rolling(window=10).max().bfill()
    # Add Rolling Mean/Std 10 (mentioned in prompt text of NHITS notebook)
    df['rolling_mean_10'] = df['y'].rolling(window=10).mean().bfill()
    df['rolling_std_10'] = df['y'].rolling(window=10).std().bfill()
    df['rolling_mean_50'] = df['y'].rolling(window=50).mean().bfill()

    # 2. Cyclic Features - REMOVED as requested
    # Weekly
    # minutes_in_week = 7 * 24 * 60
    # df['week_sin'] = np.sin(2 * np.pi * df['ds'].dt.dayofweek * 24 * 60 / minutes_in_week)
    # df['week_cos'] = np.cos(2 * np.pi * df['ds'].dt.dayofweek * 24 * 60 / minutes_in_week)

    # Daily
    # minutes_in_day = 24 * 60
    # current_minute = df['ds'].dt.hour * 60 + df['ds'].dt.minute
    # df['day_sin'] = np.sin(2 * np.pi * current_minute / minutes_in_day)
    # df['day_cos'] = np.cos(2 * np.pi * current_minute / minutes_in_day)

    # Fill NaNs
    df = df.bfill()

    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to raw input CSV')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save processed CSV')
    args = parser.parse_args()

    preprocess_data(args.input_csv, args.output_csv)
