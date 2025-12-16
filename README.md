# NHITS & GRU Training Pipeline

## Prerequisites

Google Cloud SDK<br>
Docker<br>
Python 3.11<br>

## Setup

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
```

## Configuration

Update variables in deploy_pipeline.sh

## Execution

```bash
./deploy_pipeline.sh
```

## Project Structure

```
.
├── deploy_pipeline.sh                 # Main orchestration script: builds Docker image, compiles pipeline, and submits job to Vertex AI
├── pipeline.py                        # Kubeflow Pipeline (KFP) definition: defines steps (Extract -> Preprocess -> Train) and resources (GPU)
├── submit_pipeline.py                 # Python script to submit the compiled pipeline job using the Vertex AI SDK
├── Dockerfile                         # Defines the container environment (TensorFlow, PyTorch, CUDA) used by all pipeline steps
├── requirements.txt                   # Python dependencies installed inside the Docker container
├── weather_data.csv                   # Historical weather data included in the image for feature engineering
├── src/                               # Source code for individual pipeline components
│   ├── extract.py                     # Step 1: Extracts raw data from BigQuery to CSV
│   ├── preprocess.py                  # Step 2: Cleans data, removes outliers, and merges with weather data
│   ├── train_gru.py                   # Step 3: Trains the GRU model on GPU and saves artifacts to GCS
│   ├── train_model.py                 # Alternative training script (NHITS/N-BEATS)
│   ├── evaluate_models.py             # Script for evaluating model performance metrics
│   ├── prediction_utils.py            # Helper functions for generating predictions
│   └── streaming_pipeline.py          # Logic for real-time/streaming inference
└── training_and_preprocessing_workflows/ # Jupyter notebooks for initial experimentation and analysis
    ├── nhits_training_workflow.ipynb  # Development notebook for NHITS model
    ├── tensorflow_lstm_gru_workflow.ipynb # Development notebook for GRU/LSTM models
    └── ml_dataset_preprocessing.ipynb # Data exploration and cleaning prototypes
```
