export PROJECT_ID="time-series-478616"
export BUCKET_NAME="time-series-478616-ml-pipeline"
export REPO_NAME="ml-pipelines"
export BQ_QUERY='select
  arrival_date,
  duration,
  mbt
from `mta_historical_v3.ml_cleaned`
where extract(year from arrival_date) >= 2024'


# Replace with your desired name (must be globally unique)
export BUCKET_NAME="time-series-478616-ml-pipeline"
export REGION="us-east1"
gcloud storage buckets create gs://$BUCKET_NAME --location=$REGION