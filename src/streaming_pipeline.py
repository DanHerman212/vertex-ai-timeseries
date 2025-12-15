import argparse
import logging
import datetime
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, StandardOptions
from apache_beam.io.gcp.bigtableio import WriteToBigtable
from apache_beam.transforms.userstate import BagStateSpec, ReadModifyWriteStateSpec
from apache_beam.coders import PickleCoder
import prediction_utils

# Define Bigtable configuration
PROJECT_ID = "your-project-id"
INSTANCE_ID = "your-bigtable-instance"
TABLE_ID = "predictions"

class StatefulBufferingFn(beam.DoFn):
    """
    Accumulates history for each device_id.
    Maintains a buffer of the last 150 arrivals.
    """
    # State to hold the list of arrival timestamps
    ARRIVAL_HISTORY = BagStateSpec('arrival_history', PickleCoder())
    # State to hold the last observation (to detect arrivals)
    LAST_OBSERVATION = ReadModifyWriteStateSpec('last_observation', PickleCoder())

    def process(self, element, 
                arrival_history=beam.DoFn.StateParam(ARRIVAL_HISTORY),
                last_observation=beam.DoFn.StateParam(LAST_OBSERVATION)):
        """
        element: (device_id, data_point_dict)
        data_point_dict: {'timestamp': iso_str, 'eta': float}
        """
        device_id, new_data = element
        current_time = datetime.datetime.fromisoformat(new_data['timestamp'])
        current_eta = new_data['eta']
        
        # Get last observation
        last_obs = last_observation.read()
        
        if last_obs:
            last_eta = last_obs['eta']
            last_time = datetime.datetime.fromisoformat(last_obs['timestamp'])
            
            # Check if arrival occurred
            if prediction_utils.detect_arrival(current_eta, last_eta, current_time, last_time):
                # Arrival happened roughly at last_time + last_eta (minutes)
                # Or simply assume it happened at 'last_time' if the ETA was close to 0
                # For simplicity, let's assume the arrival happened at the moment the ETA reset
                # which is roughly 'current_time' minus a small delta, or just 'current_time'.
                # A better approximation: Arrival Time = Last Time + Last ETA
                approx_arrival_time = last_time + datetime.timedelta(minutes=last_eta)
                
                # Add to history
                arrival_history.add(approx_arrival_time)
                
        # Update last observation
        last_observation.write(new_data)
        
        # Prepare input for model
        # We need to read the history to build the dataframe
        current_history = list(arrival_history.read())
        
        # If we have enough history, we can predict
        if len(current_history) >= 2:
            model_input_df = prediction_utils.prepare_model_input(
                device_id, 
                current_history, 
                target_steps=150
            )
            
            if model_input_df is not None:
                # We need the LAST confirmed arrival time to anchor our prediction
                # This is the last element in our sorted history
                sorted_history = sorted(current_history)
                last_arrival_time = sorted_history[-1]
                
                yield {
                    'device_id': device_id,
                    'input_df': model_input_df,
                    'last_arrival_time': last_arrival_time,
                    'current_poll_time': current_time
                }

class PredictDoFn(beam.DoFn):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_handler = None

    def setup(self):
        self.model_handler = prediction_utils.ModelHandler(self.model_path)

    def process(self, element):
        if element is None:
            return

        try:
            input_df = element['input_df']
            device_id = element['device_id']
            last_arrival_time = element['last_arrival_time']
            current_poll_time = element['current_poll_time']
            
            forecast_df = self.model_handler.predict(input_df)
            
            bt_rows = prediction_utils.postprocess_prediction(
                forecast_df, device_id, last_arrival_time, current_poll_time
            )
            
            for row_key, cell_data in bt_rows:
                yield (row_key, cell_data)
                
        except Exception as e:
            logging.error(f"Prediction failed: {e}")

class CreateBigtableRow(beam.DoFn):
    def process(self, element):
        row_key, cell_data = element
        from google.cloud.bigtable import row
        direct_row = row.DirectRow(row_key=row_key)
        column_family_id = 'cf1'
        
        for col_qualifier, value in cell_data.items():
            direct_row.set_cell(
                column_family_id,
                col_qualifier.encode('utf-8'),
                value.encode('utf-8')
            )
        yield direct_row

def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_topic', required=True, help='Pub/Sub topic')
    parser.add_argument('--model_path', required=True, help='GCS path to model')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--bt_instance', required=True, help='Bigtable Instance')
    parser.add_argument('--bt_table', required=True, help='Bigtable Table')
    parser.add_argument('--station_id', required=True, help='Target Station ID to process')
    
    known_args, pipeline_args = parser.parse_known_args(argv)
    
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(StandardOptions).streaming = True

    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | "ReadFromPubSub" >> beam.io.ReadFromPubSub(topic=known_args.input_topic)
            | "ParseAndFilter" >> beam.Map(lambda msg: prediction_utils.parse_target_station(msg, known_args.station_id))
            | "FilterInvalid" >> beam.Filter(lambda x: x is not None)
            # Stateful processing requires Key-Value pairs
            | "AccumulateHistory" >> beam.ParDo(StatefulBufferingFn())
            | "RunInference" >> beam.ParDo(PredictDoFn(model_path=known_args.model_path))
            | "CreateBigtableRow" >> beam.ParDo(CreateBigtableRow())
            | "WriteToBigtable" >> WriteToBigtable(
                project_id=known_args.project_id,
                instance_id=known_args.bt_instance,
                table_id=known_args.bt_table
            )
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
