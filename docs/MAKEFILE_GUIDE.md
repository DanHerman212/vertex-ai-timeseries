# Project Operations Guide (Makefile)

This project uses a `Makefile` to standardize common operational tasks such as deployment, testing, and infrastructure management. This guide explains how to use these commands.

## Prerequisites

- **Make**: Ensure `make` is installed on your system.
  - **macOS**: Pre-installed (or via Xcode Command Line Tools).
  - **Linux**: `sudo apt-get install make` (usually pre-installed).
  - **Windows**: Install via Chocolatey (`choco install make`) or use WSL.

## Available Commands

### 1. Deploy Pipeline
Deploys the full training pipeline to Vertex AI. This includes building Docker images, compiling the pipeline JSON, and submitting the job.

```bash
make deploy
```
*Under the hood: Runs `scripts/deploy_pipeline.sh`*

### 2. Run Local Tests (Python)
Runs the training and evaluation scripts locally using your current Python environment. Useful for quick debugging of code logic without Docker overhead.

```bash
make test
```
*Under the hood: Runs `scripts/test_local_workflow.sh`*

### 3. Run Local Tests (Containerized)
Builds a local test Docker image and runs the training/evaluation scripts inside a container. This ensures your code works in the production-like Linux environment (especially useful for verifying file paths and dependencies).

```bash
make test-container
```
*Under the hood: Runs `scripts/test_container_local.sh`*

### 4. Setup GCE Instance & Test Streaming
Provisions and configures a Google Compute Engine (GCE) instance for the data ingestion service. It handles VM creation, file transfer, service startup, and **automatically starts the streaming pipeline in dry-run mode** to verify connectivity.

```bash
make setup-gce
```
*Under the hood: Runs `scripts/setup_gce_and_run.sh`*

### 5. Stop GCE Services
Stops the ingestion service and the GCE instance to save costs.

```bash
make stop
```
*Under the hood: Runs `scripts/stop_vm_pipeline.sh`*
*Under the hood: Runs `scripts/test_streaming_dryrun.sh`*

### 6. Stop GCE Services
Stops the ingestion services and related processes on the GCE instance to save costs or pause data collection.

```bash
make stop
```
*Under the hood: Runs `scripts/stop_vm_pipeline.sh`*

### 6. Teardown Infrastructure
Permanently deletes the GCE instance, Pub/Sub topic, and Pub/Sub subscription. Use this when you are completely done with the project or want to reset the environment. **Warning: This action is destructive.**

```bash
make teardown
```
*Under the hood: Runs `scripts/teardown_gce.sh`*

### 7. Load Image Variables
Helper command that prints instructions on how to load Docker image environment variables into your current shell session.

```bash
make image-vars
```
*Usage: Run `source scripts/image_variables.sh` manually based on the output.*

## Environment Variables
The Makefile automatically loads variables from a `.env` file in the project root if it exists. Ensure your `.env` file is populated with the necessary GCP project details (PROJECT_ID, BUCKET_NAME, etc.).
