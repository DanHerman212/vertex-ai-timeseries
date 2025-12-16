# ML Pipelines for Time Series Forecasting on Vertex AI
## Challenger/Champion ML Workflow with NHITS and GRU Models
This repository contains a complete machine learning pipeline for time series forecasting using NHITS and GRU models, deployed on Google Cloud Vertex AI. The pipeline is built with Kubeflow Pipelines (KFP) and leverages GPU resources for efficient model training.

## Overview
The pipeline consists of the following steps:
1. **Extract**: Retrieve raw time series data from BigQuery 
2. **Preprocess**: Clean the data, remove outliers, engineer features from weather, rolling regime and trigonomic functions cos and sin
3. **Train**: Train both NHITS and GRU models using GPU
4. **Evaluate**: Assess model performance using Mean Absolute Error (MAE) on a test dataset, selecting the best model as the champion


The repo is nearly complete and is expected to finish before the end of DEC 2025.  Feel free to browse all the code and ask any questions.


## File Structure

```
.
├── deploy_pipeline.sh                 # Main orchestration script: builds Docker image, compiles pipeline, and submits job to Vertex AI
├── pipeline.py                        # Kubeflow Pipeline (KFP) definition: defines steps (Extract -> Preprocess -> Train -> Evaluate) and resources (GPU)
├── submit_pipeline.py                 # Python script to submit the compiled pipeline job using the Vertex AI SDK
├── Dockerfile                         # Defines the container environment (TensorFlow, PyTorch, CUDA) used by all pipeline steps
├── requirements.txt                   # Python dependencies installed inside the Docker container
├── weather_data.csv                   # Historical weather data included in the image for feature engineering
├── src/                               # Source code for individual pipeline components
│   ├── extract.py                     # Step 1: Extracts raw data from BigQuery to CSV
│   ├── preprocess.py                  # Step 2: Cleans data, removes outliers, and merges with weather data
│   ├── train_gru.py                   # Step 3: Trains the GRU model on GPU and saves artifacts to GCS
│   ├── train_model.py                 # Alternative training script (NHITS/N-BEATS)
│   ├── evaluate_models.py             # Step 4: Evaluates model performance (MAE) on test set
│   ├── prediction_utils.py            # Helper functions for generating predictions
│   └── streaming_pipeline.py          # Logic for real-time/streaming inference
└── training_and_preprocessing_workflows/ # Jupyter notebooks for initial experimentation and analysis
    ├── nhits_training_workflow.ipynb  # Development notebook for NHITS model
    ├── tensorflow_lstm_gru_workflow.ipynb # Development notebook for GRU/LSTM models
    └── ml_dataset_preprocessing.ipynb # Data exploration and cleaning prototypes
```
