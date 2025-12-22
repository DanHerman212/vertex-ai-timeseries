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

## Phase 2: Test Deployment (Automated)

We use a `Makefile` to automate the deployment of the ingestion service to a GCE instance and run the pipeline in "dry-run" mode to verify logic.

### Step 1: Setup & Run
Run the following command to:
1.  Create/Start the GCE instance.
2.  Deploy the Ingestion Service (Producer).
3.  Run the Streaming Pipeline (Consumer) in dry-run mode.

```bash
make setup-gce
```

### Step 2: Verify Output
The command will stream logs to your terminal. Look for:
*   `Received message with X entities`
*   `Found match: Route E...`
*   `Calculated duration: ...`

---

## Phase 3: Production Deployment (Cloud)

Once Phase 2 is verified:

### Step 1: Deploy Pipeline to Dataflow
Submit the job to Dataflow runner using the automated script:

```bash
make deploy-streaming
```

This will:
1.  Build the pipeline package.
2.  Submit a job to Google Cloud Dataflow.
3.  Run continuously, scaling workers as needed.
