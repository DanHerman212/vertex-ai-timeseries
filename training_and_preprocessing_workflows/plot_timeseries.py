import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_zoomed_slice(df, col_name, start_idx=0, end_idx=400):
    """
    Plots a specific slice of the dataset with dynamic date labels.
    """
    subset = df.iloc[start_idx:end_idx]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(subset[col_name])
    ax.set_xlabel('Time (Train Sequence)')
    ax.set_ylabel(col_name)
    
    # Generate dynamic ticks based on the actual data in the slice
    tick_indices = np.linspace(start_idx, end_idx-1, 8, dtype=int)
    
    # Ensure 'date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['arrival_date']):
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        
    tick_labels = df['arrival_date'].iloc[tick_indices].dt.strftime('%Y-%m-%d %H:%M')
    
    plt.xticks(tick_indices, tick_labels)
    plt.xlim(start_idx, end_idx)
    
    fig.autofmt_xdate()
    plt.title(f'Zoomed View: {tick_labels.iloc[0]} to {tick_labels.iloc[-1]}')
    plt.tight_layout()
    plt.show()

def plot_heatmap(df, value_col='mbt', timezone='America/New_York'):
    """
    Plots a heatmap of average values by Hour vs Day of Week.
    """
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['arrival_date']):
        df['date'] = pd.to_datetime(df['arrival_date'])

    # Create temporary features for plotting
    plot_df = df.copy()
    
    # Convert to local timezone if data is in UTC
    if plot_df['arrival_date'].dt.tz is not None:
        plot_df['arrival_date'] = plot_df['arrival_date'].dt.tz_convert(timezone)
    
    plot_df['hour'] = plot_df['arrival_date'].dt.hour
    plot_df['day_name'] = plot_df['arrival_date'].dt.day_name()

    pivot_table = plot_df.pivot_table(
        values=value_col, 
        index='hour', 
        columns='day_name', 
        aggfunc='mean'
    )

    # Reorder days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    existing_days = [day for day in days_order if day in pivot_table.columns]
    pivot_table = pivot_table[existing_days]

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='viridis_r', annot=True, fmt=".1f", linewidths=.5)
    
    plt.title(f'Average {value_col} (Heatmap)')
    plt.ylabel('Hour of Day')
    plt.xlabel('Day of Week')
    plt.tight_layout()
    plt.show()

def plot_distribution(df, col_name):
    """
    Plots the histogram distribution of a column.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col_name], bins=50, kde=True)
    plt.title(f'Distribution of {col_name}')
    plt.xlabel(col_name)
    plt.show()

def plot_rolling_average(df, col_name, window_size=50):
    """
    Plots a rolling average of the specified column to smooth out noise.
    """
    # Calculate rolling mean
    smooth_data = df[col_name].rolling(window=window_size).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(smooth_data, color='red', linewidth=2)

    ax.set_title(f'{window_size}-Train Rolling Average of {col_name}')
    ax.set_ylabel(f'Average {col_name}')
    plt.tight_layout()
    plt.show()