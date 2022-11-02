#! /bin/bash

IMAGE_NAME=robocar-oak-camera
TAG=$(git describe)
FULL_IMAGE_NAME=git.cyrilix.bzh/robocars/${IMAGE_NAME}:${TAG}
PLATFORM="linux/amd64,linux/arm64"
#PLATFORM="linux/amd64,linux/arm64,linux/arm/v7"

podman build . --platform "${PLATFORM}" --manifest "${IMAGE_NAME}:${TAG}"
podman manifest push --all "localhost/${IMAGE_NAME}:${TAG}" "docker://${FULL_IMAGE_NAME}"

printf "\nImage %s published" "docker://${FULL_IMAGE_NAME}"
