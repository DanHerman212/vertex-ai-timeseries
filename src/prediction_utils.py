import pandas as pd
import numpy as np
import logging
from neuralforecast import NeuralForecast
import json
import datetime

class ModelHandler:
    """
    Singleton-like class to handle model loading and inference.
    """
    _instance = None
    _model = None

    def __new__(cls, model_path=None):
        if cls._instance is None:
            cls._instance = super(ModelHandler, cls).__new__(cls)
            if model_path:
                cls._instance._load_model(model_path)
        return cls._instance

    def _load_model(self, model_path):
        logging.info(f"Loading model from {model_path}...")
        self._model = NeuralForecast.load(path=model_path)
        logging.info("Model loaded successfully.")

    def predict(self, input_df):
        if self._model is None:
            raise ValueError("Model not initialized. Call with model_path first.")
        return self._model.predict(df=input_df)

def parse_target_station(message_data, target_station_id):
    """
    Parses the message and extracts data for the specific station.
    Handles single objects, lists of objects, or keyed dictionaries.
    """
    try:
        if isinstance(message_data, bytes):
            message_data = message_data.decode('utf-8')
        
        data = json.loads(message_data)
        
        found_item = None
        
        # Strategy 1: Top-level match
        if isinstance(data, dict):
            if data.get('device_id') == target_station_id or data.get('station_id') == target_station_id:
                found_item = data
            # Strategy 2: Keyed by station_id
            elif target_station_id in data:
                found_item = data[target_station_id]
            # Strategy 3: 'entity' list (GTFS style) or just 'data' list
            elif 'entity' in data and isinstance(data['entity'], list):
                for item in data['entity']:
                    if item.get('device_id') == target_station_id or item.get('station_id') == target_station_id:
                        found_item = item
                        break
        
        # Strategy 4: List of objects
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and (item.get('device_id') == target_station_id or item.get('station_id') == target_station_id):
                    found_item = item
                    break
        
        if found_item:
            timestamp = found_item.get('timestamp')
            value = found_item.get('minutes_until_arrival')
            
            if timestamp and value is not None:
                return (target_station_id, {
                    'timestamp': timestamp, 
                    'eta': float(value)
                })
                
        return None

    except Exception as e:
        logging.error(f"Error parsing message: {e}")
        return None

def detect_arrival(current_eta, last_eta, current_time, last_time):
    """
    Heuristic to detect if a train arrived between the last poll and this one.
    Logic: If ETA dropped significantly (reset) or we crossed the 0-bound.
    
    Simple check: If current ETA is significantly larger than last ETA (e.g. 1 min -> 15 min),
    it implies the previous train arrived and we are now tracking the next one.
    """
    if last_eta is None:
        return False
        
    # Thresholds
    # If ETA jumped up by more than 3 minutes, it's likely a reset
    if current_eta > last_eta + 3.0:
        return True
        
    return False

def prepare_model_input(device_id, arrival_history, target_steps=150):
    """
    Constructs the input DataFrame based on arrival history (Headways).
    
    Args:
        device_id: The ID of the series.
        arrival_history: List of datetime objects representing arrival times.
        target_steps: The number of past arrivals to include.
    """
    if len(arrival_history) < 2:
        return None

    # Sort just in case
    arrival_history = sorted(arrival_history)
    
    # Calculate headways (difference between arrivals in minutes)
    # y[t] = Arrival[t] - Arrival[t-1]
    df_data = []
    for i in range(1, len(arrival_history)):
        current_arrival = arrival_history[i]
        prev_arrival = arrival_history[i-1]
        diff = (current_arrival - prev_arrival).total_seconds() / 60.0
        
        # Filter out unrealistic headways (e.g. < 1 min might be noise or duplicate detection)
        if diff < 1.0: 
            continue
            
        df_data.append({
            'ds': current_arrival, # The time the train arrived
            'y': diff              # The headway (gap) that just finished
        })
    
    df = pd.DataFrame(df_data)
    
    # Keep only the last N steps
    df = df.tail(target_steps)
    
    if df.empty:
        return None
        
    df['unique_id'] = device_id
    return df

def postprocess_prediction(prediction_df, device_id, last_arrival_time, current_poll_time):
    """
    Converts the predicted headway into a 'minutes remaining' forecast.
    """
    rows = []
    
    # prediction_df contains the forecast for the NEXT step(s).
    # Usually just one row if horizon=1
    for index, row in prediction_df.iterrows():
        # Get predicted headway
        pred_col = [c for c in row.index if c not in ['ds', 'unique_id', 'y']][0]
        predicted_headway_minutes = float(row[pred_col])
        
        # Calculate Predicted Arrival Time
        # Next Arrival = Last Arrival + Predicted Headway
        predicted_arrival_time = last_arrival_time + datetime.timedelta(minutes=predicted_headway_minutes)
        
        # Calculate Minutes Remaining from NOW
        minutes_remaining = (predicted_arrival_time - current_poll_time).total_seconds() / 60.0
        
        # Row Key: device_id#current_timestamp (so app can query 'latest')
        # Or device_id#predicted_arrival_time
        # Let's use current_timestamp so we have a history of predictions made at T
        row_key = f"{device_id}#{current_poll_time.isoformat()}".encode('utf-8')
        
        cell_data = {
            'predicted_minutes_remaining': str(minutes_remaining),
            'predicted_arrival_time': predicted_arrival_time.isoformat(),
            'predicted_headway': str(predicted_headway_minutes),
            'last_arrival_time': last_arrival_time.isoformat(),
            'generated_at': datetime.datetime.now().isoformat()
        }
        rows.append((row_key, cell_data))
        
    return rows
