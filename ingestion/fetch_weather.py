import requests
import os
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, location="10022"):
        self.api_key = api_key
        self.location = location
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

    def fetch_current_weather(self):
        """
        Fetches the current weather and hourly forecast for the location.
        """
        if not self.api_key:
            logging.error("Visual Crossing API Key is missing.")
            return None

        # Guidance URL format
        # https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{zipcode}/today
        # Parameters: unitGroup=us, key=..., contentType=json, include=hours, elements=...
        
        base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        elements = "datetime,temp,precip,snow,snowdepth,visibility,windspeed"
        
        url = f"{base_url}/{self.location}/today?unitGroup=us&key={self.api_key}&contentType=json&include=hours&elements={elements}"
        
        print(f"Testing URL: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching weather data: {e}")
            if response.status_code == 400:
                 logging.error(f"Response: {response.text}")
            return None

    def parse_weather_data(self, data):
        """
        Parses the JSON response to extract relevant fields for the model.
        Returns a dictionary keyed by datetime (hour).
        """
        if not data:
            return {}

        parsed_data = {}
        
        # Process hourly data
        if 'days' in data and len(data['days']) > 0:
            day = data['days'][0]
            date_str = day['datetime'] # YYYY-MM-DD
            
            for hour in day.get('hours', []):
                time_str = hour['datetime'] # HH:MM:SS
                full_dt_str = f"{date_str}T{time_str}"
                
                # Extract features matching model requirements
                features = {
                    'temp': hour.get('temp'),
                    'precip': hour.get('precip', 0.0),
                    'snow': hour.get('snow', 0.0),
                    'snowdepth': hour.get('snowdepth', 0.0),
                    'visibility': hour.get('visibility'),
                    'windspeed': hour.get('windspeed')
                }
                
                # Handle None values
                for k, v in features.items():
                    if v is None:
                        features[k] = 0.0 if k in ['precip', 'snow', 'snowdepth'] else 0.0 # Default for others?
                
                parsed_data[full_dt_str] = features
                
        return parsed_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test execution
    # key = os.environ.get("VISUAL_CROSSING_KEY")
    key = "LCV5P97B2K8LWEBUV93DD9BW9"
    if key:
        fetcher = WeatherFetcher(key)
        data = fetcher.fetch_current_weather()
        if data:
            print(f"Resolved Address: {data.get('resolvedAddress')}")
            print(f"Requested Address: {data.get('address')}")
        parsed = fetcher.parse_weather_data(data)
        print(json.dumps(parsed, indent=2))
    else:
        print("Please set VISUAL_CROSSING_KEY environment variable to test.")
