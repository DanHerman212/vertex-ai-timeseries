#!/bin/bash
set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
else
    echo "Error: .env file not found."
    exit 1
fi

MODEL_NAME="nhits-forecast-model"
# Ensure REGION is set
REGION=${REGION:-"us-east1"}

echo "========================================================"
echo "CLEANUP: Removing old model versions"
echo "========================================================"

# 1. Get Model ID
echo "Finding Model ID for $MODEL_NAME..."
MODEL_ID_FULL=$(gcloud ai models list --region=$REGION --filter="display_name=$MODEL_NAME" --format="value(name)" | head -n 1)
# MODEL_ID_FULL is like projects/123/locations/us-east1/models/456
MODEL_ID=${MODEL_ID_FULL##*/}

if [ -z "$MODEL_ID" ]; then
    echo "Error: Model not found."
    exit 1
fi
echo "Model ID: $MODEL_ID"

# 2. Get Latest Version
echo "Identifying latest version..."
# Get version ID with max createTime
LATEST_VERSION_ID=$(gcloud ai models list-version $MODEL_ID --region=$REGION --format="json" | python3 -c "import sys, json; print(max([int(v['versionId']) for v in json.load(sys.stdin)]))")

echo "Latest Version is: $LATEST_VERSION_ID"
echo "Keeping version $LATEST_VERSION_ID. All others will be deleted."

# 3. Check Endpoint for Deployed Old Versions
if [ -n "$ENDPOINT_ID" ]; then
    echo "Checking Endpoint $ENDPOINT_ID for old deployments..."
    
    # Get list of deployed models: id, model (full path with version)
    # We use a python snippet to generate the undeploy commands safely
    gcloud ai endpoints describe $ENDPOINT_ID --region=$REGION --format="json" | python3 -c "
import sys, json, subprocess

try:
    endpoint_json = json.load(sys.stdin)
    deployed_models = endpoint_json.get('deployedModels', [])
    target_model_id = '$MODEL_ID'
    keep_version = '$LATEST_VERSION_ID'
    endpoint_id = '$ENDPOINT_ID'
    region = '$REGION'
    project_id = '$PROJECT_ID'

    for dm in deployed_models:
        # model field looks like: projects/.../models/MODEL_ID@VERSION_ID
        model_ref = dm.get('model', '')
        deployed_id = dm.get('id')
        
        if target_model_id in model_ref:
            # Extract version
            try:
                if '@' in model_ref:
                    version = model_ref.split('@')[-1]
                else:
                    # If no @, it's version 1 usually, or the default. 
                    # We assume if it doesn't match our keep_version logic we might want to remove it,
                    # but be careful. For now, assume explicit versioning.
                    version = '1' 

                if version != keep_version:
                    print(f'Undeploying old version {version} (Deployed ID: {deployed_id})...')
                    cmd = [
                        'gcloud', 'ai', 'endpoints', 'undeploy-model', endpoint_id,
                        '--project', project_id,
                        '--region', region,
                        '--deployed-model-id', deployed_id,
                        '--quiet'
                    ]
                    subprocess.run(cmd, check=True)
                else:
                    print(f'Skipping active version {version} (Deployed ID: {deployed_id})')
            except Exception as e:
                print(f'Error processing {model_ref}: {e}')
except Exception as e:
    print(f'Error parsing endpoint details: {e}')
"
fi

# 4. Delete Old Versions
echo "Deleting old versions from Registry..."
# List all versions
VERSIONS=$(gcloud ai models list-version $MODEL_ID --region=$REGION --format="value(versionId)")

for v in $VERSIONS; do
    if [ "$v" != "$LATEST_VERSION_ID" ]; then
        echo "Deleting version $v..."
        gcloud ai models delete-version $MODEL_ID --version-id=$v --region=$REGION --quiet || echo "Warning: Failed to delete version $v (might still be undeploying or in use)"
    fi
done

echo "Cleanup Complete!"
