# Deep Learning Workspace

This document provides an overview of the Jupyter notebooks available in the `training_and_preprocessing_workflows` directory. These notebooks cover the end-to-end workflow from data preprocessing to training advanced deep learning models for subway headway prediction.

## Notebooks

### 1. [Data Preprocessing](training_and_preprocessing_workflows/ml_dataset_preprocessing.ipynb)
**Description:** 
This notebook focuses on the initial stages of the machine learning pipeline. It handles:
- Environment setup and configuration (including Google Cloud Storage authentication).
- Loading and cleaning the raw dataset.
- Initial data visualization and time series plotting.
- Preprocessing steps required before model training.

### 2. [N-HiTS Training Workflow](training_and_preprocessing_workflows/nhits_training_workflow.ipynb)
**Description:** 
A specialized workflow for training the **N-HiTS (Neural Hierarchical Interpolation for Time Series)** model using the `neuralforecast` library.
- **Goal:** Predict subway headway (Minutes Between Trains - MBT).
- **Features:** Utilizes exogenous variables (weather), endogenous features (rolling means/stds), and cyclic features (weekly/yearly seasonality).
- **Process:** Covers data preparation, model initialization, training, and evaluation.

### 3. [TensorFlow LSTM/GRU Workflow](training_and_preprocessing_workflows/tensorflow_lstm_gru_workflow.ipynb)
**Description:** 
A comprehensive guide to training standard deep learning architectures using **TensorFlow/Keras**.
- **Models Covered:** Dense (Baseline), LSTM (Long Short-Term Memory), and GRU (Gated Recurrent Unit).
- **Key Steps:** 
    - Data scaling and normalization.
    - Creation of windowed datasets for time series forecasting.
    - Training and comparing the performance of different model architectures.
