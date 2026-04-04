#!/bin/bash
#
# Start a Docker container for the privacy pipeline.
#
# Adjust the paths and GPU device below to match your setup.
#
# Usage:
#   ./run-container.sh              # default: GPU 0
#   ./run-container.sh 1            # use GPU 1
#   ./run-container.sh all          # use all GPUs

set -e

CONTAINER_NAME="privacy_pipeline"
IMAGE_NAME="privacy-pipeline:latest"
GPU_DEVICE="${1:-0}"

# ---------------------------------------------------------------------------
# Paths: adjust these to your local directory layout.
#   DATA_DIR    -> raw images and pipeline intermediate outputs
#   PIPELINE_DIR -> this repository (scripts, src, config)
#   OUTPUT_DIR  -> trained models, anonymised images, evaluation results
# ---------------------------------------------------------------------------
DATA_DIR="${HOME}/data"
PIPELINE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${HOME}/output"

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

echo "GPU device : ${GPU_DEVICE}"
echo "Data       : ${DATA_DIR}  -> /workspace/data"
echo "Pipeline   : ${PIPELINE_DIR}  -> /workspace/privacy_pipeline"
echo "Output     : ${OUTPUT_DIR}  -> /workspace/output"
echo ""

if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    echo "Container '${CONTAINER_NAME}' already exists."
    echo "  Start:  docker start -i ${CONTAINER_NAME}"
    echo "  Remove: docker rm ${CONTAINER_NAME}"
    exit 1
fi

if [ "${GPU_DEVICE}" = "all" ]; then
    GPU_FLAG="--gpus all"
else
    GPU_FLAG="--gpus '\"device=${GPU_DEVICE}\"'"
fi

eval docker run -d \
    --name "${CONTAINER_NAME}" \
    ${GPU_FLAG} \
    --shm-size=16g \
    -v "${DATA_DIR}":/workspace/data \
    -v "${PIPELINE_DIR}":/workspace/privacy_pipeline \
    -v "${OUTPUT_DIR}":/workspace/output \
    "${IMAGE_NAME}" \
    sleep infinity

echo ""
echo "Container '${CONTAINER_NAME}' is running."
echo ""
echo "Enter the container:"
echo "  docker exec -it ${CONTAINER_NAME} /bin/bash"
echo ""
echo "Stop:"
echo "  docker stop ${CONTAINER_NAME}"
