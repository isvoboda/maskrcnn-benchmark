#%/usr/bin/env bash

# Default user is John Wick bro.
user="${1:-jw}"

docker run \
    -it \
    --rm \
    -v "$(pwd):/app" \
    ${user}-maskrcnn-benchmark-dev:1.1 \

