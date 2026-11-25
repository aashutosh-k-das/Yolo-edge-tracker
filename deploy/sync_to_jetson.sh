#!/usr/bin/env bash
# Run on your LAPTOP: copy the code + ONNX model(s) + class names to the Jetson.
#   bash deploy/sync_to_jetson.sh <user>@<jetson-ip> [remote_dir]
# Copies .onnx, NOT .engine: engines only work on the GPU + TensorRT version that built them,
# so build them on the Jetson (python3 -m export.build_engine ...).
set -euo pipefail
HOST=${1:?usage: sync_to_jetson.sh user@jetson-ip [remote_dir]}
DEST=${2:-~/yolo-edge-tracker}
cd "$(dirname "$0")/.."
rsync -avz --progress \
    --include 'weights/' --include 'weights/*.onnx' --exclude 'weights/*' \
    --include 'data/' --include 'data/dataset.yaml' --exclude 'data/*' \
    --exclude 'runs/' --exclude 'benchmark/results/' --exclude '.git/' --exclude '__pycache__/' \
    --exclude '*.mp4' --exclude '*.engine' \
    ./ "$HOST:$DEST/"
echo
echo "Next, on the Jetson:  cd $DEST && bash deploy/jetson_setup.sh"
