#!/usr/bin/env bash
# Phase 3 in one go, on a machine with an NVIDIA GPU + TensorRT (Colab T4, RTX desktop, or Jetson*):
#   PyTorch -> ONNX -> TensorRT FP16 + INT8 -> verify all -> benchmark all
#
#   bash export/run_all.sh runs/detect/yolo_custom/weights/best.pt data/dataset/images/test [calib_dir]
#
# *Jetson Nano (Python 3.6, no Ultralytics): export ONNX elsewhere, copy weights/<name>.onnx over,
#  and run only the build_engine + benchmark lines below.
set -euo pipefail
PT=$1
IMAGES=$2
CALIB=${3:-data/dataset/images/train}   # calibrate on TRAIN images, evaluate on TEST
NAME=$(python3 -c "from export.export_onnx import default_name; print(default_name('$PT'))")
W=weights/$NAME

python3 -m export.export_onnx --weights "$PT" --images "$IMAGES"
python3 -m export.build_engine --onnx "$W.onnx" --fp16
python3 -m export.build_engine --onnx "$W.onnx" --int8 --calib-images "$CALIB"
python3 -m export.verify --reference "$W.pt" --images "$IMAGES" \
    --models "$W.onnx" "${W}_fp16.engine" "${W}_int8.engine"
python3 -m benchmark.benchmark --images "$IMAGES" --plot --weights \
    "$W.pt" "$W.pt@fp16" "$W.onnx" "${W}_fp16.engine" "${W}_int8.engine"
echo "Accuracy of every variant: python3 -m benchmark.eval_accuracy --weights $W.pt $W.onnx ${W}_fp16.engine ${W}_int8.engine"
