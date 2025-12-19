import requests
import logging
import pandas as pd
from datetime import datetime, timedelta

class WeatherFetcher:
    def __init__(self, api_key, location="10022"): # 10022 is NYC Zip
        self.api_key = api_key
        self.location = location
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        self.cache = {}
        self.last_fetch_time = None
        self.fetch_interval = timedelta(minutes=30) # Refresh every 30 mins

    def fetch_current_weather(self):
        """
        Fetches the current weather and hourly forecast for the location.
        """
        if not self.api_key:
            return None

        # Guidance URL format
        # https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{zipcode}/today
        # Parameters: unitGroup=us, key=..., contentType=json, include=hours, elements=...
        
        base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        elements = "datetime,temp,precip,snow,snowdepth,visibility,windspeed"
        
        # Use 'next24hours' for streaming to ensure we have future coverage for predictions
        # But stick to the guidance parameters for efficiency
        url = f"{base_url}/{self.location}/next24hours?unitGroup=us&key={self.api_key}&contentType=json&include=hours&elements={elements}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching weather data: {e}")
            return None

    def update_cache(self):
        """
        Updates the internal cache with fresh data.
        """
        now = datetime.utcnow()
        if self.last_fetch_time and (now - self.last_fetch_time) < self.fetch_interval:
            return

        logging.info("Fetching fresh weather data...")
        data = self.fetch_current_weather()
        if not data:
            return

        # Parse and update cache
        if 'days' in data:
            for day in data['days']:
                date_str = day['datetime']
                for hour in day.get('hours', []):
                    time_str = hour['datetime']
                    # Visual Crossing returns HH:MM:SS
                    dt_str = f"{date_str} {time_str}"
                    try:
                        dt = pd.to_datetime(dt_str)
                        # Round to nearest hour just in case
                        dt_hour = dt.round('h')
                        
                        features = {
                            'temp': hour.get('temp'),
                            'precip': hour.get('precip', 0.0),
                            'snow': hour.get('snow', 0.0),
                            'snowdepth': hour.get('snowdepth', 0.0),
                            'visibility': hour.get('visibility'),
                            'windspeed': hour.get('windspeed')
                        }
                        
                        # Handle None
                        for k, v in features.items():
                            if v is None:
                                features[k] = 0.0
                                
                        self.cache[dt_hour] = features
                    except Exception as e:
                        logging.warning(f"Error parsing weather hour: {e}")

        self.last_fetch_time = now
        logging.info(f"Weather cache updated. {len(self.cache)} entries.")

    def get_features(self, timestamp):
        """
        Get features for a specific timestamp (datetime object).
        """
        # Ensure cache is populated
        self.update_cache()
        
        # Lookup
        ts_hour = timestamp.round('h')
        if ts_hour.tz is not None:
            ts_hour = ts_hour.tz_localize(None)
            
        return self.cache.get(ts_hour)
