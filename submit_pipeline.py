from google.cloud import aiplatform
import argparse
import sys

def submit_pipeline(project_id, region, bucket_name, pipeline_root, pipeline_json, bq_query, training_image_uri):
    print(f"Initializing Vertex AI SDK for project {project_id} in {region}...")
    aiplatform.init(project=project_id, location=region, staging_bucket=bucket_name)

    print(f"Submitting pipeline job from {pipeline_json}...")
    job = aiplatform.PipelineJob(
        display_name="gru-training-pipeline",
        template_path=pipeline_json,
        pipeline_root=pipeline_root,
        parameter_values={
            "project_id": project_id,
            "bq_query": bq_query,
            "bucket_name": bucket_name,
            "training_image_uri": training_image_uri
        },
        enable_caching=True
    )

    job.submit()
    print(f"Pipeline submitted successfully!")
    print(f"Dashboard URL: {job._dashboard_uri()}")
    print(f"Job Resource Name: {job.resource_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--bucket_name', type=str, required=True)
    parser.add_argument('--pipeline_root', type=str, required=True)
    parser.add_argument('--pipeline_json', type=str, required=True)
    parser.add_argument('--bq_query', type=str, required=True)
    parser.add_argument('--training_image_uri', type=str, required=True)
    
    args = parser.parse_args()
    
    try:
        submit_pipeline(
            args.project_id,
            args.region,
            args.bucket_name,
            args.pipeline_root,
            args.pipeline_json,
            args.bq_query,
            args.training_image_uri
        )
    except Exception as e:
        print(f"Error submitting pipeline: {e}")
        sys.exit(1)
