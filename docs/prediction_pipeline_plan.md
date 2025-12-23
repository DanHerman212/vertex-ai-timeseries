# Prediction Pipeline Implementation Plan

## Overview
This document outlines the plan to implement a real-time prediction pipeline that ingests JSON messages, transforms them, queries the NHITS model, and stores the results in a NoSQL database.

## Phase 1: Requirements & Design
-  **Analyze Input Data**: Review sample JSON messages to understand the schema and required transformations.
-  **Define Output Schema**: Determine the structure for the NoSQL database (Firestore).
-  **Architecture Confirmation**: Confirm the use of Pub/Sub -> Dataflow -> Vertex AI -> Firestore.

## Phase 2: Data Ingestion & Transformation
-  **Input Source**: Configure the pipeline to read from the message source (e.g., Pub/Sub subscription).
-  **Parsing Logic**: Implement parsing of the specific JSON message format (GTFS-Realtime).
-  **Feature Engineering**: Transform raw message data into the format expected by the NHITS model (time series windows).

## Phase 3: Model Integration
-  **Vertex AI Client**: Implement the logic to make batch/online prediction requests to the deployed Vertex AI Endpoint.
- **Error Handling**: Handle model timeouts, retries, and fallback logic.

## Phase 4: Storage Layer
-  **Database Setup**: Provision the Firestore collection.
-  **Write Logic**: Implement the sink to write predictions and metadata to the database.

## Phase 5: Deployment & Testing
-  **Local Testing**: Run the pipeline locally with the sample JSONs.
-  **Deployment**: Deploy the Dataflow job to GCP.
-  **Validation**: Verify end-to-end flow from message injection to database record.
