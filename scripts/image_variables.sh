# 1. Load your project variables (PROJECT_ID, REPO_NAME)
export $(cat .env | grep -v '#' | awk '/=/ {print $1}')

# 2. Define the tag of the EXISTING images
OLD_TAG="v20251221-213400"
REPO=${REPO_NAME:-ml-pipelines} # Default to ml-pipelines if not set

# 3. Set the URIs for the images you want to SKIP building
# export TENSORFLOW_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO}/tensorflow-training:${OLD_TAG}"

export PYTORCH_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO}/pytorch-serving:${OLD_TAG}"

export PYTORCH_SERVING_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${PYTORCH_IMAGE_NAME}:${OLD_TAG}"

# 4. Ensure the one you want to BUILD is unset (so it generates a new tag/URI)
unset TENSORFLOW_IMAGE_URI
