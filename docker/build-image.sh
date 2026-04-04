#!/bin/bash
#
# Build the Docker image for the privacy pipeline.
#
# Usage:
#   ./build-image.sh
#   ./build-image.sh --no-cache

set -e

IMAGE_NAME="privacy-pipeline"
VERSION="latest"

echo "Building Docker image: ${IMAGE_NAME}:${VERSION}"

docker build ${1:---progress=plain} -t "${IMAGE_NAME}:${VERSION}" .

echo ""
echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${VERSION}"
echo ""
echo "Next step:"
echo "  ./run-container.sh"
