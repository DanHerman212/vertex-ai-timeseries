#!/bin/bash
set -e

# Hardcoded for safety based on your context
MODEL_ID="1862568299406032896"
REGION="us-east1"
KEEP_VERSION="10"

echo "========================================================"
echo "MANUAL CLEANUP: Deleting versions 1-9, KEEPING v$KEEP_VERSION"
echo "========================================================"

# Loop through versions 1 to 9
for v in {1..9}; do
    echo "Deleting version $v..."
    # We allow this to fail (e.g. if already deleted) without stopping the script
    gcloud ai models delete-version $MODEL_ID --version-id=$v --region=$REGION --quiet || echo "Version $v might already be gone."
done

echo "========================================================"
echo "Cleanup Complete. Only Version $KEEP_VERSION should remain."
echo "========================================================"
