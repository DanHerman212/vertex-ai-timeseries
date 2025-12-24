.PHONY: deploy test stop build-local setup-gce

# Load environment variables if .env exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Deploy the pipeline to Vertex AI for ML training
deploy:
	@echo "🚀 Deploying Pipeline..."
	@chmod +x scripts/deploy_pipeline.sh
	@./scripts/deploy_pipeline.sh


# Setup and run the GCE instance for ingestion
setup-gce:
	@echo "☁️ Setting up GCE Instance..."
	@chmod +x scripts/setup_gce_and_run.sh
	@./scripts/setup_gce_and_run.sh

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

# Deploy the ingestion service to GCE (Production)
deploy-ingestion:
	@echo "☁️ Deploying Ingestion Service to GCE..."
	@chmod +x scripts/deploy_ingestion.sh
	@./scripts/deploy_ingestion.sh

# Deploy the streaming pipeline to Dataflow (Production)
deploy-streaming:
	@echo "🌊 Deploying Streaming Pipeline to Dataflow..."
	@chmod +x scripts/deploy_streaming_production.sh
	@./scripts/deploy_streaming_production.sh

# Helper to source image variables (prints instructions as make runs in a subshell)
image-vars:
	@echo "ℹ️  To load image variables, run this in your shell:"
	@echo "source scripts/image_variables.sh"
