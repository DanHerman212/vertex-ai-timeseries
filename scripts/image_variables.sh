
export $(cat .env | grep -v '#' | awk '/=/ {print $1}')


OLD_TAG="v20251221-213400"
REPO=${REPO_NAME:-ml-pipelines} # Default to ml-pipelines if not set

export PYTORCH_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO}/pytorch-training:${OLD_TAG}"

export PYTORCH_SERVING_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO}/pytorch-serving:${OLD_TAG}"

old_tf_tag="v20251221-225104"
export TENSORFLOW_IMAGE_URI="us-east1-docker.pkg.dev/${PROJECT_ID}/${REPO}/tensorflow-training:${old_tf_tag}"
