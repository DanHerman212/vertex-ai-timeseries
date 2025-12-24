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
        
        # --- PADDING LOGIC (Ensure 160 rows) ---
        required_history = 160
        current_history = len(df)
        prediction_status = "LIVE"
        
        if current_history < required_history:
            prediction_status = f"WARMUP ({current_history}/{required_history})"
            missing_count = required_history - current_history
            
            # Calculate defaults for padding
            avg_mbt = df['y'].mean() if not df.empty else 5.0
            avg_duration = df['duration'].mean() if not df.empty else 25.0
            first_ds = df['ds'].iloc[0] if not df.empty else pd.Timestamp.now()
            
            # Generate padding rows (going backwards in time)
            padding_data = []
            for i in range(1, missing_count + 1):
                # Go back i intervals (approximate)
                prev_ts = first_ds - pd.Timedelta(minutes=avg_mbt * i)
                padding_data.append({
                    'ds': prev_ts,
                    'y': avg_mbt,
                    'duration': avg_duration,
                    'unique_id': key.split('_')[0]
                })
            
            # Prepend padding (reverse list so oldest is first)
            padding_df = pd.DataFrame(padding_data[::-1])
            df = pd.concat([padding_df, df], ignore_index=True)
            
            logging.info(f"⚠️  Padded input with {missing_count} synthetic rows. Status: {prediction_status}")
        else:
            logging.info(f"✅ Input history complete ({current_history} rows). Status: {prediction_status}")

        # 2. Feature Engineering (Rolling Stats)
        # Note: We use min_periods=1 to ensure we get values even at the start of the window
        df['rolling_mean_10'] = df['y'].rolling(window=10, min_periods=1).mean()
        df['rolling_std_10'] = df['y'].rolling(window=10, min_periods=1).std().fillna(0)
        df['rolling_max_10'] = df['y'].rolling(window=10, min_periods=1).max()
        df['rolling_mean_50'] = df['y'].rolling(window=50, min_periods=1).mean()
        df['rolling_std_50'] = df['y'].rolling(window=50, min_periods=1).std().fillna(0)
        
        # 3. Append Future Row (for Exogenous Features)
        # The model needs to know the weather/calendar for the NEXT timestamp to predict it.
        last_ts = df['ds'].iloc[-1]
        # We assume the next arrival is roughly the average MBT away, or just 15 mins.
        # Since these are exogenous features (weather/day of week), exact minute doesn't matter too much.
        future_ts = last_ts + pd.Timedelta(minutes=15)
        
        # Create a future row with NaNs for target/rolling, but valid unique_id and ds
        future_row = {
            'ds': future_ts,
            'unique_id': key.split('_')[0],
            'y': 0.0, # Dummy
            'duration': 0.0, # Dummy
            'rolling_mean_10': 0.0, 'rolling_std_10': 0.0, 'rolling_max_10': 0.0,
            'rolling_mean_50': 0.0, 'rolling_std_50': 0.0
        }
        
        # Append to DF temporarily to process weather in one go (or just append to list)
        # Actually, let's just append it to the dataframe
        df = pd.concat([df, pd.DataFrame([future_row])], ignore_index=True)
        
        # 4. Weather & Calendar Features (for ALL rows, including future)
        # This is inefficient to do row-by-row for 161 rows every time, but safe.
        # Optimization: Vectorize this later.
        
        weather_data = []
        for ts in df['ds']:
            feats = self.get_weather_features(ts)
            # Add Calendar features
            feats['dow'] = 1 if ts.weekday() >= 5 else 0 # 1=Weekend, 0=Weekday
            weather_data.append(feats)
            
        weather_df = pd.DataFrame(weather_data)
        df = pd.concat([df, weather_df], axis=1)
        
        # 5. Prepare Instances for Vertex AI
        # Convert timestamps to strings for JSON serialization
        df['ds'] = df['ds'].astype(str)
        
        # Select only necessary columns (to reduce payload size)
        required_cols = [
            'ds', 'unique_id', 'y', 'duration',
            'rolling_mean_10', 'rolling_std_10', 'rolling_max_10', 
            'rolling_mean_50', 'rolling_std_50',
            'temp', 'precip', 'snow', 'snowdepth', 'visibility', 'windspeed', 'dow'
        ]
        
        # Filter columns that actually exist
        final_cols = [c for c in required_cols if c in df.columns]
        
        # CRITICAL FIX: Handle NaNs and Infinity which cause 400 Bad Request in JSON
        df_final = df[final_cols].copy()
        df_final = df_final.fillna(0)
        df_final = df_final.replace([np.inf, -np.inf], 0)
        
        instances = df_final.to_dict(orient='records')

        if self.dry_run:
            logging.info(f"DRY RUN: Generated {len(instances)} instances for {key}")
            if instances:
                # Log the FUTURE row (last one)
                latest = instances[-1]
                msg = (
                    f"\n🔍 FUTURE VECTOR SNAPSHOT ({key}):\n"
                    f"   📅 Target Time: {latest.get('ds')}\n"
                    f"   🌤️  Weather: Temp={latest.get('temp')}°F, Wind={latest.get('windspeed')} mph\n"
                    f"   📅 DOW: {latest.get('dow')}"
                )
                logging.info(msg)
            
            yield {
                'key': key,
                'input_last_timestamp': element['last_timestamp'],
                'prediction_status': prediction_status,
                'forecast': [0.0], 
                'dry_run_features': instances
            }
            return

        try:
            # 6. Call Endpoint
            prediction = self.endpoint.predict(instances=instances)
            forecast = prediction.predictions
            
            logging.info(f"🔮 PREDICTION RECEIVED for {key}: {forecast}")

            yield {
                'key': key,
                'input_last_timestamp': element['last_timestamp'],
                'prediction_status': prediction_status,
                'forecast': forecast
            }
            
        except Exception as e:
            logging.error(f"Error calling Vertex AI for {key}: {e}")
            # Log the payload to help debug 400 errors
            logging.error(f"Payload that caused error (first 2 records): {instances[:2]}")
            logging.error(f"Full payload length: {len(instances)}")

