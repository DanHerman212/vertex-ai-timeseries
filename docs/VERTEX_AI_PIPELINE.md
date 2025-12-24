# Vertex AI ML Pipeline Workflow

This document details the automated Machine Learning pipeline implemented in `pipeline.py`. The pipeline is orchestrated using **Vertex AI Pipelines** (Kubeflow) and handles the end-to-end workflow from data extraction to model deployment.

## Pipeline Workflow

![Vertex AI Pipeline](../images/vertexaipipelines.png)

## Pipeline Steps

The pipeline consists of the following sequential and parallel steps:

### 1. Data Extraction (`extract_bq_data`)
*   **Input**: BigQuery SQL Query.
*   **Action**: Executes the query against the BigQuery dataset to retrieve historical subway arrival data.
*   **Output**: A CSV dataset artifact (`output_dataset`).

### 2. Data Preprocessing (`preprocess_component`)
*   **Input**: Raw CSV dataset from Step 1.
*   **Action**: 
    *   Cleans the data.
    *   Integrates external weather data (temperature, precipitation, etc.).
    *   Formats timestamps and columns for training.
*   **Output**: A processed CSV dataset (`output_csv`).

### 3. Model Training (Parallel Branches)

The pipeline splits into two parallel branches to train different model architectures.

#### Branch A: GRU Model (TensorFlow)
*   **Step 3a: Train GRU (`train_gru_component`)**
    *   Trains a Gated Recurrent Unit (GRU) neural network using TensorFlow/Keras.
    *   Uses GPU acceleration (`NVIDIA_TESLA_T4`).
    *   **Output**: SavedModel directory (`model_dir`) and a test dataset.
*   **Step 4a: Upload GRU Model (`ModelUploadOp`)**
    *   Uploads the trained GRU model to the **Vertex AI Model Registry**.
    *   Attaches the serving container image URI (`tf2-cpu.2-15`).

#### Branch B: N-HiTS Model (NeuralForecast)
*   **Step 3b: Train N-HiTS (`train_nhits_component`)**
    *   Trains the N-HiTS (Neural Hierarchical Interpolation for Time Series) model.
    *   Uses GPU acceleration (`NVIDIA_TESLA_T4`).
    *   **Output**: PyTorch checkpoint (`model_dir`), forecast dataframe, and logs.
*   **Step 4b: Upload N-HiTS Model (`ModelUploadOp`)**
    *   Uploads the trained N-HiTS model to the **Vertex AI Model Registry**.
    *   Attaches a custom serving container image URI.

### 4. Model Evaluation
Both models are evaluated on a hold-out test set to generate performance metrics.

*   **Evaluate GRU (`evaluate_gru_component`)**: Generates loss plots and prediction visualizations.
*   **Evaluate N-HiTS (`evaluate_nhits_component`)**: Calculates MAE, MSE, and generates forecast plots.

### 5. Deployment (Automated)

Once the N-HiTS model is successfully trained and uploaded, the pipeline automatically deploys it.

*   **Step 7: Create Endpoint (`EndpointCreateOp`)**
    *   Creates a new Vertex AI Endpoint named `nhits-endpoint`.
*   **Step 8: Deploy Model (`ModelDeployOp`)**
    *   Deploys the trained N-HiTS model to the newly created endpoint.
    *   **Configuration**:
        *   Machine Type: `n1-standard-2`
        *   Replica Count: 1
        *   Traffic Split: 100%

## Artifacts & Outputs

*   **Models**: Stored in Vertex AI Model Registry.
*   **Endpoints**: Active REST API for real-time predictions.
*   **Metrics**: Visualized in the Vertex AI Pipelines UI (Confusion matrices, Loss curves).
*   **Logs**: Training logs available in Cloud Logging.
