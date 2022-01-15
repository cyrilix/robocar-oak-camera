#! /bin/bash

IMAGE_NAME=robocar-oak-camera
TAG=$(git describe)
FULL_IMAGE_NAME=docker.io/cyrilix/${IMAGE_NAME}:${TAG}


podman build . --platform linux/amd64,linux/arm64,linux/arm/v7 --manifest ${IMAGE_NAME}
podman manifest push --format v2s2 --rm "localhost/${IMAGE_NAME}" "docker://${FULL_IMAGE_NAME}"
