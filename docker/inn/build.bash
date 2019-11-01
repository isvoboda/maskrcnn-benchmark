#%/usr/bin/env bash

# Default user is John Wick bro.
user="${1:-jw}"

DOCKER_BUILDKIT=1 docker build \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    --target dev-image \
    --ssh default \
    --add-host=git.ba.innovatrics.net:"$(getent hosts git.ba.innovatrics.net | cut -d' ' -f1)" \
    -t ${user}-maskrcnn-benchmark-dev:1.0 \
    -f Dockerfile ../../
