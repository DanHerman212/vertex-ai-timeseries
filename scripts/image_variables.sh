
export $(cat .env | grep -v '#' | awk '/=/ {print $1}')


# OLD_TAG="v20251221-225104"
REPO=${REPO_NAME:-ml-pipelines} # Default to ml-pipelines if not set

old_pytorch_tag="v20251221-225104"
export PYTORCH_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO}/pytorch-training:${old_pytorch_tag}"

old_pytorch_serving_tag="v20251221-225104"
export PYTORCH_SERVING_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO}/pytorch-serving:${old_pytorch_serving_tag}"

old_tf_tag="v20251221-225104"
export TENSORFLOW_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO}/tensorflow-training:${old_tf_tag}"
