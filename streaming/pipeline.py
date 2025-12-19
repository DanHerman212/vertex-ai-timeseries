import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import argparse
from streaming.transform import ParseVehicleUpdates, CalculateTripDuration, AccumulateArrivals
from streaming.prediction import VertexAIPrediction
from streaming.sink import WriteToFirestore

def run(argv=None):
    """
    Main entry point for the streaming pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_subscription',
        help='Input Pub/Sub subscription to read from.')
    parser.add_argument(
        '--input_file_pattern',
        help='Path to local JSON files for testing (e.g., streaming/json_files/*.json).')
    parser.add_argument(
        '--output_collection',
        default='predictions',
        help='Firestore collection to write results to.')
    parser.add_argument(
        '--project_id',
        required=True,
        help='GCP Project ID.')
    parser.add_argument(
        '--region',
        default='us-east1',
        help='GCP Region.')
    parser.add_argument(
        '--endpoint_id',
        required=True,
        help='Vertex AI Endpoint ID.')
    parser.add_argument(
        '--weather_csv',
        default='weather_data.csv',
        help='Path to weather data CSV for exogenous features.')
    parser.add_argument(
        '--weather_api_key',
        help='Visual Crossing API Key for live weather data.')
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='If set, skips Vertex AI prediction and Firestore write.')
    
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(beam.options.pipeline_options.StandardOptions).streaming = True

    with beam.Pipeline(options=pipeline_options) as p:
        
        # 1. Read Input (Pub/Sub or Files)
        if known_args.input_subscription:
            messages = (p 
                | "ReadFromPubSub" >> beam.io.ReadFromPubSub(subscription=known_args.input_subscription)
            )
        elif known_args.input_file_pattern:
            # For testing with local files
            messages = (p 
                | "ReadFromFiles" >> beam.io.ReadFromText(known_args.input_file_pattern)
            )
        else:
            raise ValueError("Either --input_subscription or --input_file_pattern must be provided.")

        # 2. Parse and Filter
        # Yields (trip_id, update_dict)
        parsed_updates = (messages
            | "ParseVehicleUpdates" >> beam.ParDo(ParseVehicleUpdates(target_route_id="E", origin_stop_id="G05S", target_stop_id="F11S"))
        )

        # 3. Calculate Duration
        # Yields (route_stop_key, update_with_duration)
        durations = (parsed_updates
            | "CalculateTripDuration" >> beam.ParDo(CalculateTripDuration(origin_stop_id="G05S", target_stop_id="F11S"))
        )

        # 4. Accumulate History (Stateful)
        # Yields {'key': key, 'timestamps': [...], 'durations': [...], ...}
        windows = (durations
            | "AccumulateArrivals" >> beam.ParDo(AccumulateArrivals())
        )

        # 5. Predict with Vertex AI
        predictions = (windows
            | "VertexAIPrediction" >> beam.ParDo(VertexAIPrediction(
                project_id=known_args.project_id,
                region=known_args.region,
                endpoint_id=known_args.endpoint_id,
                weather_csv_path=known_args.weather_csv,
                weather_api_key=known_args.weather_api_key,
                dry_run=known_args.dry_run
            ))
        )

        # 6. Write to Firestore
        if not known_args.dry_run:
            (predictions
                | "WriteToFirestore" >> beam.ParDo(WriteToFirestore(
                    project_id=known_args.project_id,
                    collection_name=known_args.output_collection
                ))
            )
        else:
            # In dry run, just log the predictions
            (predictions
                | "LogPredictions" >> beam.Map(lambda x: logging.info(f"Dry Run Prediction: {x}"))
            )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
