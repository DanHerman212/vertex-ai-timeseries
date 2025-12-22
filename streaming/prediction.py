import logging
import apache_beam as beam
import pandas as pd
import numpy as np
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from streaming.weather import WeatherFetcher

class VertexAIPrediction(beam.DoFn):
    """
    Takes a window of timestamps, calculates MBT (Minutes Between Trains),
    generates exogenous features (rolling stats, calendar, weather),
    and calls the Vertex AI Endpoint for a forecast.
    """
    def __init__(self, project_id, region, endpoint_id, weather_csv_path=None, weather_api_key=None, dry_run=False):
        self.project_id = project_id
        self.region = region
        self.endpoint_id = endpoint_id
        self.weather_csv_path = weather_csv_path
        self.weather_api_key = weather_api_key
        self.dry_run = dry_run
        self.client = None
        self.weather_df = None
        self.weather_fetcher = None
        self.weather_df = None

    def setup(self):
        """
        Initialize the Vertex AI client and load weather data once per worker.
        """
        if not self.dry_run:
            aiplatform.init(project=self.project_id, location=self.region)
            self.endpoint = aiplatform.Endpoint(self.endpoint_id)
        
        # Initialize Weather Fetcher if API key is present
        if self.weather_api_key:
            self.weather_fetcher = WeatherFetcher(self.weather_api_key)
            logging.debug("WeatherFetcher initialized.")
        
        # Load weather data if provided (Fallback)
        if self.weather_csv_path:
            try:
                # In a real Dataflow job, this file needs to be accessible (e.g., in GCS or copied to worker)
                # For local testing, local path works.
                self.weather_df = pd.read_csv(self.weather_csv_path)
                if 'datetime' in self.weather_df.columns:
                    self.weather_df['datetime'] = pd.to_datetime(self.weather_df['datetime'])
                    self.weather_df = self.weather_df.sort_values('datetime')
                    # Set index for fast lookup
                    self.weather_df.set_index('datetime', inplace=True)
                logging.debug("Weather data loaded successfully.")
            except Exception as e:
                logging.warning(f"Could not load weather data: {e}. Using defaults.")

    def get_weather_features(self, timestamp):
        """
        Look up weather features for a given timestamp.
        """
        features = {
            'temp': 0.0, 'precip': 0.0, 'snow': 0.0, 
            'snowdepth': 0.0, 'visibility': 10.0, 'windspeed': 0.0
        }
        
        # 1. Try Live Fetcher
        if self.weather_fetcher:
            try:
                live_features = self.weather_fetcher.get_features(timestamp)
                if live_features:
                    return live_features
            except Exception as e:
                logging.warning(f"Live weather fetch failed: {e}")

        # 2. Fallback to CSV
        if self.weather_df is not None:
            try:
                # Round to nearest hour for lookup
                ts_hour = timestamp.round('h')
                
                # Ensure naive datetime if index is naive
                if self.weather_df.index.tz is None and ts_hour.tz is not None:
                    ts_hour = ts_hour.tz_localize(None)
                
                # Use 'asof' to find the nearest prior/current weather record
                # This handles forward filling automatically
                idx = self.weather_df.index.asof(ts_hour)
                
                if idx is not None:
                    record = self.weather_df.loc[idx]
                    for col in features.keys():
                        if col in record:
                            features[col] = float(record[col])
            except Exception as e:
                # Fallback to defaults on error
                pass
                
        return features

    def process(self, element):
        """
        element: {
            'key': 'Route_Stop',
            'timestamps': [ts1, ts2, ...],
            'durations': [d1, d2, ...],
            'last_timestamp': ts_last
        }
        """
        key = element['key']
        timestamps = sorted(element['timestamps'])
        durations = element.get('durations', [])
        
        if len(timestamps) < 2:
            logging.warning(f"Not enough timestamps to calculate MBT for {key}")
            return

        # 1. Calculate MBT (Minutes Between Trains)
        data = []
        # We need to align durations with timestamps. 
        # Assuming durations[i] corresponds to timestamps[i]
        # MBT at i is (timestamps[i] - timestamps[i-1])
        
        for i in range(1, len(timestamps)):
            t_curr = timestamps[i]
            t_prev = timestamps[i-1]
            
            diff_seconds = t_curr - t_prev
            mbt = diff_seconds / 60.0
            
            # Get duration for the current arrival if available
            curr_duration = durations[i] if i < len(durations) else 0.0
            
            data.append({
                'ds': pd.to_datetime(t_curr, unit='s'),
                'y': mbt,
                'duration': curr_duration,
                'unique_id': key.split('_')[0]
            })
            
        df = pd.DataFrame(data)
        
        # 2. Feature Engineering
        
        # A. Rolling Features (Endogenous)
        # Note: We use min_periods=1 to ensure we get values even at the start of the window
        df['rolling_mean_10'] = df['y'].rolling(window=10, min_periods=1).mean()
        df['rolling_std_10'] = df['y'].rolling(window=10, min_periods=1).std().fillna(0)
        df['rolling_max_10'] = df['y'].rolling(window=10, min_periods=1).max()
        df['rolling_mean_50'] = df['y'].rolling(window=50, min_periods=1).mean()
        df['rolling_std_50'] = df['y'].rolling(window=50, min_periods=1).std().fillna(0)
        
        # B. Cyclic Features (Calendar) - Commented out as per user request
        # minutes_in_week = 7 * 24 * 60
        # minutes_in_day = 24 * 60
        
        # Ensure ds is datetime
        df['ds'] = pd.to_datetime(df['ds'])
        
        # df['week_sin'] = np.sin(2 * np.pi * df['ds'].dt.dayofweek * 24 * 60 / minutes_in_week)
        # df['week_cos'] = np.cos(2 * np.pi * df['ds'].dt.dayofweek * 24 * 60 / minutes_in_week)
        
        # current_minute = df['ds'].dt.hour * 60 + df['ds'].dt.minute
        # df['day_sin'] = np.sin(2 * np.pi * current_minute / minutes_in_day)
        # df['day_cos'] = np.cos(2 * np.pi * current_minute / minutes_in_day)
        
        # Day of Week (0=Weekday, 1=Weekend)
        df['dow'] = (df['ds'].dt.dayofweek >= 5).astype(int)
        
        # C. Weather Features (Exogenous)
        weather_cols = ['temp', 'precip', 'snow', 'snowdepth', 'visibility', 'windspeed']
        
        # Apply weather lookup row by row (can be optimized, but fine for streaming batch sizes)
        weather_data = df['ds'].apply(self.get_weather_features).apply(pd.Series)
        df = pd.concat([df, weather_data], axis=1)
        
        # 3. Prepare Request
        # Convert datetime back to string for JSON serialization
        df['ds'] = df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert DataFrame to list of dicts
        instances = df.to_dict(orient='records')
        
        if self.dry_run:
            logging.info(f"DRY RUN: Generated {len(instances)} instances for {key}")
            if instances:
                logging.info(f"Sample Feature Vector: {instances[-1]}")
            # Yield a dummy prediction to keep the pipeline flowing if needed
            yield {
                'key': key,
                'input_last_timestamp': element['last_timestamp'],
                'forecast': [0.0] * 10, # Dummy forecast
                'dry_run_features': instances
            }
            return

        try:
            # 4. Call Endpoint
            prediction = self.endpoint.predict(instances=instances)
            forecast = prediction.predictions
            
            yield {
                'key': key,
                'input_last_timestamp': element['last_timestamp'],
                'forecast': forecast
            }
            
        except Exception as e:
            logging.error(f"Error calling Vertex AI for {key}: {e}")

