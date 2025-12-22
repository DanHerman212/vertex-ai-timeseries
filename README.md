# ML Pipelines for Time Series Forecasting on Vertex AI
## Challenger/Champion ML Workflow with NHITS and GRU Models
This repo includes end to end machine learning pipelines for time series forecasting on Vertex AI.<br>  

The forecasting task centers around the public transit domain for the NYC subway system.  The target forecast is minutes until the next train arrives, at a particular station, to help passengers manage uncertainty in planning their commute.<br>

The data representation from the subway system is generated from censors on subway vehicles, at high frequency with irregular intervals.  Given the complexity of the data, we train two different models with distinct architectures to gain perspective on what's possible with deep learning. <br>

The first model is a traditional RNN using a stacked and regularlized GRU architecture with TensorFlow and Keras.  The second is a N-HiTS model, which is a state of the art multi-stack MLP, that uses hierarchical interpolation and multi-rate sampling to handle different timescales, deployed with NeuralForecast. <br><br> The pipeline is designed to run on Vertex AI using Kubeflow Pipelines (KFP) and leverages GPU acceleration for model training.

<div align="center">
  <img src="images/vertexaipipelines.png" alt="Image Description">
</div>

For detailed instructions on running commands, see the [Project Operations Guide](docs/MAKEFILE_GUIDE.md).

## Overview
The pipeline consists of the following steps:
1. **Extract**: Retrieve time series data from BigQuery 
2. **Preprocess**: Clean the data, remove outliers, engineer features from weather, rolling regime and trigonomic functions cos and sin
3. **Train**: Train both NHITS and GRU models using GPU
4. **Evaluate**: Assess model performance using Mean Absolute Error (MAE) on a test dataset, selecting the best model as the champion
5. **Attach Serving Spec**: Prepare the winning model for deployment with appropriate serving specifications
6. **Deploy**: Deploy the champion model to Vertex AI Model Registry for online predictions


The repo is nearly complete and is expected to finish before the end of DEC 2025.  Feel free to browse all the code and ask any questions.


## File Structure

```
nhits_gcp_workflow/
├── Makefile                            # Main entry point for running tasks (deploy, test, etc.)
├── pipeline.py                         # Definition of the Vertex AI pipeline structure
├── README.md                           # Project overview and instructions
├── submit_pipeline.py                  # Script to submit the pipeline job to Vertex AI
├── weather_data.csv                    # Dataset used for training/testing
├── docker/                             # Docker configuration files
│   ├── Dockerfile                      # Default Dockerfile for the project
│   ├── Dockerfile.local_test           # Dockerfile for local testing
│   ├── Dockerfile.nhits                # Dockerfile specifically for the NHITS model environment
│   └── Dockerfile.serving              # Dockerfile for the model serving container
├── docs/                               # Documentation files
│   ├── deep_learning_workspace.md
│   ├── deployment_blockers_summary.md
│   ├── deployment_guide.md
│   ├── prediction_pipeline_plan.md
│   └── STREAMING_GUIDE.md
├── images/                             # Images used in documentation
│   ├── forecasting_pipeline.png
│   ├── pipelines.png
│   └── vertexaipipelines.png
├── ingestion/                          # Data ingestion scripts
│   ├── ingest_feed.py
│   ├── mta-ingestion.service
│   └── requirements.txt
├── requirements/                       # Python dependency files
│   ├── requirements.nhits.txt          # Dependencies for NHITS training
│   └── requirements.txt                # General project dependencies
├── scripts/                            # Operational scripts
│   ├── deploy_pipeline.sh              # Script to deploy the Vertex AI pipeline
│   ├── image_variables.sh              # Environment variables for Docker images
│   ├── setup_gce_and_run.sh            # Script to setup GCE instance
│   ├── stop_vm_pipeline.sh             # Script to stop GCE services
│   ├── test_container_local.sh         # Script to run containerized tests
│   └── test_local_workflow.sh          # Script to run local python tests
├── src/                                # Source code for pipeline components
│   ├── evaluate_gru.py                 # Component for evaluating GRU model
│   ├── evaluate_nhits.py               # Component for evaluating NHITS model
│   ├── extract.py                      # Component for data extraction
│   ├── prediction_utils.py             # Utility functions for making predictions
│   ├── preprocess.py                   # Component for data preprocessing
│   ├── serve.py                        # Code for serving the model
│   ├── train_gru.py                    # Training script for the GRU model
│   └── train_nhits.py                  # Training script for the NHITS model
├── streaming/                          # Streaming pipeline code
│   ├── pipeline.py
│   ├── prediction.py
│   ├── sink.py
│   ├── test_transform.py
│   └── transform.py
└── training_and_preprocessing_workflows/ # Jupyter notebooks for experimentation
    ├── ml_dataset_preprocessing.ipynb
    ├── model_utils.py
    ├── nhits_training_workflow.ipynb
    ├── plot_timeseries.py
    └── tensorflow_lstm_gru_workflow.ipynb
```
