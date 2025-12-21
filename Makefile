.PHONY: deploy test stop build-local setup-gce

# Load environment variables if .env exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Deploy the pipeline to Vertex AI
deploy:
	@echo "🚀 Deploying Pipeline..."
	@chmod +x scripts/deploy_pipeline.sh
	@./scripts/deploy_pipeline.sh

# Run the local workflow test (Python based)
test:
	@echo "🧪 Running Local Workflow Test..."
	@chmod +x scripts/test_local_workflow.sh
	@./scripts/test_local_workflow.sh

# Run the local container test (Docker based)
test-container:
	@echo "🐳 Running Local Container Test..."
	@chmod +x scripts/test_container_local.sh
	@./scripts/test_container_local.sh

# Setup and run the GCE instance for ingestion
setup-gce:
	@echo "☁️ Setting up GCE Instance..."
	@chmod +x scripts/setup_gce_and_run.sh
	@./scripts/setup_gce_and_run.sh

# Run the streaming pipeline in dry-run mode on the GCE instance
test-streaming:
	@echo "🌊 Running Streaming Pipeline (Dry Run)..."
	@chmod +x scripts/test_streaming_dryrun.sh
	@./scripts/test_streaming_dryrun.sh

# Stop the GCE instance and services
stop:
	@echo "🛑 Stopping GCE Services..."
	@chmod +x scripts/stop_vm_pipeline.sh
	@./scripts/stop_vm_pipeline.sh

# Teardown GCE instance and Pub/Sub resources
teardown:
	@echo "🗑️  Tearing down GCE and Pub/Sub resources..."
	@chmod +x scripts/teardown_gce.sh
	@./scripts/teardown_gce.sh

# Helper to source image variables (prints instructions as make runs in a subshell)
image-vars:
	@echo "ℹ️  To load image variables, run this in your shell:"
	@echo "source scripts/image_variables.sh"
