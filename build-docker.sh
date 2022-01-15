#! /bin/bash

IMAGE_NAME=robocar-oak-camera
TAG=$(git describe)
FULL_IMAGE_NAME=docker.io/cyrilix/${IMAGE_NAME}:${TAG}


podman build . --platform linux/amd64,linux/arm64,linux/arm/v7 --manifest "${IMAGE_NAME}:${TAG}"
podman manifest push --format v2s2 "localhost/${IMAGE_NAME}:${TAG}" "docker://${FULL_IMAGE_NAME}"

printf "\nImage %s published" "docker://${FULL_IMAGE_NAME}"
