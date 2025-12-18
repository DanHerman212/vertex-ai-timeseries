# Streaming Pipeline: End-to-End Deployment Guide

This document outlines the architecture and step-by-step deployment process for the Real-Time Prediction Pipeline.

## Architecture Overview

1.  **Ingestion (GCE Instance)**:
    *   Polls MTA GTFS-Realtime Feed (ACE Line) every 30 seconds.
    *   Parses Protobuf to JSON.
    *   Publishes to Pub/Sub Topic `vehicle-position-updates`.
2.  **Messaging (Pub/Sub)**:
    *   Topic: `vehicle-position-updates`
    *   Subscription: `vehicle-position-updates-sub` (Pull)
3.  **Processing (Dataflow / Apache Beam)**:
    *   Reads from Pub/Sub.
    *   **Windowing**: Accumulates 150 timestamps per Route/Stop key.
    *   **Feature Engineering**: Calculates MBT, Rolling Stats, Cyclic Time, Weather.
    *   **Inference**: Calls Vertex AI Endpoint (NHITS Model).
    *   **Sink**: Writes predictions to Firestore.

---

## Phase 1: Infrastructure Setup

Before running any code, we need the GCP resources.

1.  **Run the Setup Script**:
    ```bash
    ./setup_infrastructure.sh
    ```
    *   Enables APIs (Pub/Sub, Dataflow, Firestore, Vertex AI).
    *   Creates Pub/Sub Topic and Subscription.

2.  **Verify Firestore**:
    *   Ensure a Firestore database (Native mode) is created in your project.

---

## Phase 2: Test Deployment (Local)

We will run the components locally to verify logic before deploying to the cloud.

### Step 1: Test Ingestion (Feed -> Pub/Sub)
Run the ingestion script locally to start pumping real data into the topic.

1.  **Install Dependencies**:
    ```bash
    pip install -r ingestion/requirements.txt
    ```
2.  **Run Script**:
    ```bash
    export PROJECT_ID="time-series-478616"
    export TOPIC_ID="vehicle-position-updates"
    python ingestion/ingest_feed.py
    ```
3.  **Verify**: Check the Pub/Sub topic in GCP Console to see messages arriving.

### Step 2: Test Processing (Pub/Sub -> Pipeline -> Firestore)
Run the Beam pipeline locally (DirectRunner) to process the live stream.

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Pipeline**:
    ```bash
    python streaming/pipeline.py \
      --project_id "time-series-478616" \
      --region "us-east1" \
      --input_subscription "projects/time-series-478616/subscriptions/vehicle-position-updates-sub" \
      --endpoint_id "YOUR_VERTEX_ENDPOINT_ID" \
      --weather_csv "weather_data.csv"
    ```
    *(Note: You need the actual `endpoint_id` from your deployed model)*

---

## Phase 3: Production Deployment (Cloud)

Once Phase 2 is verified:

### Step 1: Deploy Ingestion to GCE
1.  Provision `e2-micro` VM.
2.  Copy `ingestion/` folder.
3.  Install systemd service (`ingestion/mta-ingestion.service`).

### Step 2: Deploy Pipeline to Dataflow
1.  Submit the job to Dataflow runner.
    ```bash
    python streaming/pipeline.py \
      --runner DataflowRunner \
      --project "time-series-478616" \
      --region "us-east1" \
      --temp_location "gs://YOUR_BUCKET/temp" \
      --staging_location "gs://YOUR_BUCKET/staging" \
      --input_subscription "projects/time-series-478616/subscriptions/vehicle-position-updates-sub" \
      ...
    ```
