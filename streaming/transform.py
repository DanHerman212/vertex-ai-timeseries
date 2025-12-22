import json
import logging
import apache_beam as beam
from apache_beam.transforms.userstate import BagStateSpec, ReadModifyWriteStateSpec
from apache_beam.coders import PickleCoder
from datetime import datetime

class ParseVehicleUpdates(beam.DoFn):
    """
    Parses raw JSON messages from the subway feed and extracts relevant vehicle updates.
    Filters for specific route and stop IDs (origin and target).
    """
    def __init__(self, target_route_id="E", origin_stop_id="G05S", target_stop_id="F11S"):
        self.target_route_id = target_route_id
        self.origin_stop_id = origin_stop_id
        self.target_stop_id = target_stop_id

    def process(self, element):
        """
        element: Raw JSON string or bytes.
        """
        try:
            if isinstance(element, bytes):
                element = element.decode('utf-8')
            
            data = json.loads(element)
            
            # Check if 'entity' list exists
            if 'entity' not in data:
                return

            entity_count = len(data.get('entity', []))
            # logging.info(f"Received message with {entity_count} entities")

            for entity in data['entity']:
                # We check both 'vehicle' and 'trip_update' for arrival times
                # For simplicity, let's assume we use 'vehicle' updates for now as per original code
                # But we might need to look at trip_update for future predictions if needed.
                # Sticking to original logic of looking at 'vehicle' for actual positions.
                
                if 'vehicle' not in entity:
                    continue
                
                vehicle = entity['vehicle']
                trip = vehicle.get('trip', {})
                
                route_id = trip.get('route_id')
                stop_id = vehicle.get('stop_id')
                
                # Filter for target route and EITHER origin OR target stop
                if route_id == self.target_route_id and (stop_id == self.origin_stop_id or stop_id == self.target_stop_id):
                    timestamp = vehicle.get('timestamp')
                    
                    # If timestamp is missing in vehicle, try header
                    if not timestamp:
                        timestamp = data.get('header', {}).get('timestamp')

                    if timestamp:
                        # Key by trip_id to correlate origin and target visits
                        trip_id = trip.get('trip_id')
                        if trip_id:
                            logging.info(f"Found match: Route {route_id}, Stop {stop_id}, Trip {trip_id}")
                            yield (trip_id, {
                                'route_id': route_id,
                                'stop_id': stop_id,
                                'timestamp': int(timestamp),
                                'trip_id': trip_id,
                                'status': vehicle.get('current_status')
                            })

        except json.JSONDecodeError:
            logging.error("Failed to decode JSON message")
        except Exception as e:
            logging.error(f"Error processing message: {e}")

class CalculateTripDuration(beam.DoFn):
    """
    Stateful DoFn that calculates the duration between origin and target stops for a trip.
    Keys by trip_id.
    """
    ORIGIN_TS = ReadModifyWriteStateSpec('origin_ts', PickleCoder())

    def __init__(self, origin_stop_id="G05S", target_stop_id="F11S"):
        self.origin_stop_id = origin_stop_id
        self.target_stop_id = target_stop_id

    def process(self, element, origin_ts_state=beam.DoFn.StateParam(ORIGIN_TS)):
        """
        element: (trip_id, update_dict)
        """
        trip_id, update = element
        stop_id = update['stop_id']
        timestamp = update['timestamp']
        route_id = update['route_id']

        if stop_id == self.origin_stop_id:
            # Store origin timestamp
            origin_ts_state.write(timestamp)
        
        elif stop_id == self.target_stop_id:
            # Check if we have an origin timestamp
            start_ts = origin_ts_state.read()
            if start_ts:
                duration_seconds = timestamp - start_ts
                duration_minutes = duration_seconds / 60.0
                
                # Only emit valid positive durations
                if duration_minutes > 0:
                    # Key by route_stop for the next aggregation step
                    key = f"{route_id}_{stop_id}"
                    logging.info(f"Calculated duration: {duration_minutes:.2f} min for Trip {trip_id}")
                    yield (key, {
                        'timestamp': timestamp,
                        'duration': duration_minutes
                    })
                
                # Optional: Clear state after successful match? 
                # Or keep it in case of updates? 
                # For streaming, clearing helps memory, but duplicate updates might happen.
                # Let's keep it for now or rely on TTL (not implemented here).

class AccumulateArrivals(beam.DoFn):
    """
    Stateful DoFn that accumulates the last 150 arrival timestamps and durations.
    """
    # State to hold the list of (timestamp, duration) tuples
    HISTORY = BagStateSpec('history', PickleCoder())

    def process(self, element, history_state=beam.DoFn.StateParam(HISTORY)):
        """
        element: (key, update_dict)
        update_dict: {'timestamp': ts, 'duration': dur}
        """
        key, update = element
        new_ts = update['timestamp']
        new_dur = update['duration']
        
        # Read current history
        current_history = list(history_state.read())
        
        # Deduplicate based on timestamp
        # current_history is list of dicts or tuples. Let's use dicts.
        # Check if timestamp already exists
        existing_timestamps = {item['timestamp'] for item in current_history}
        
        if new_ts not in existing_timestamps:
            current_history.append({'timestamp': new_ts, 'duration': new_dur})
            
            # Sort by timestamp
            current_history.sort(key=lambda x: x['timestamp'])
            
            # Keep only the last 160
            if len(current_history) > 160:
                current_history = current_history[-160:]
            
            # Clear and rewrite state
            history_state.clear()
            for item in current_history:
                history_state.add(item)
            
            # Emit window if we have enough data (e.g., > 10)
            if len(current_history) >= 10:
                yield {
                    'key': key,
                    'timestamps': [x['timestamp'] for x in current_history],
                    'durations': [x['duration'] for x in current_history],
                    'last_timestamp': current_history[-1]['timestamp']
                }
