# Build guide: phase by phase

The step-by-step guide used to build this project. The project overview is in the [README](../README.md).

Real-time detection and multi-object tracking of **drones, vehicles and aircraft** on edge hardware:
YOLOv8n/s → ONNX → TensorRT (FP16/INT8) → ByteTrack, deployed on a Jetson Nano with a Streamlit demo UI.

```
yolo-edge-tracker/
├── data/        prepare_dataset.py (validate, stats, leak-free split), dataset.yaml
├── train/       train.py, evaluate.py                         (GPU / Colab)
├── export/      export_onnx.py, build_engine.py, calibrator.py (ONNX, TensorRT FP16/INT8)
├── track/       byte_tracker.py, kalman_filter.py, matching.py (standalone ByteTrack, numpy+scipy)
├── deploy/      detectors.py (pt/onnx/trt), pipeline.py, run.py, jetson_setup.sh
├── benchmark/   benchmark.py (latency/FPS), eval_accuracy.py (mAP), report.py (table + plot)
├── tests/       unit tests for tracker + post-processing
├── weights/     trained / exported models
└── app.py       Streamlit UI
```

**Design choice:** everything that runs on the Nano (`track/`, `deploy/`, `export/build_engine.py`,
`benchmark/benchmark.py`) needs **only numpy, scipy, OpenCV, TensorRT and pycuda** and is
**Python 3.6-compatible**. JetPack 4.6 ships Python 3.6, which Ultralytics does not support.
Your own pre/post-processing (letterbox, YOLOv8 decoding, NMS) is implemented in `deploy/`.

---

## Phase 0: Environment

```bash
# dev machine / Colab
conda create -n yolo-edge python=3.11 -y && conda activate yolo-edge
pip install -r requirements.txt
pytest                       # tracker + post-processing tests should pass
```

Hardware roles:
| Where | What |
|---|---|
| Colab / Kaggle / lab GPU | training, ONNX export, mAP evaluation |
| Laptop (CPU) | ONNX Runtime testing, Streamlit demo |
| Jetson Nano | TensorRT engine build, real-time inference, speed benchmark |

## Phase 1: Dataset

Full walkthrough: **[data/ROBOFLOW.md](../data/ROBOFLOW.md)** (labelling rules, Roboflow steps, export, augmentation).

1. **Collect** video clips and images of each class (public sources: Anti-UAV, Drone-vs-Bird, VisDrone for vehicles, FGVC-Aircraft and similar). Note the licence of each source for your report.
2. **Annotate** in Roboflow (or CVAT / LabelImg) and export in **YOLOv8 format**. Class ids are alphabetical: `0 aircraft, 1 drone, 2 vehicle`.
   Aim for ≥1,500 boxes per class, and include ~5-10% background images (sky, birds, clouds) to cut false positives.
3. **Validate + split:**
   ```bash
   python data/prepare_dataset.py --roboflow data/raw/roboflow --dry-run                 # stats only
   python data/prepare_dataset.py --roboflow data/raw/roboflow --group-pattern "^(.*)_\d+$"
   # non-Roboflow export:
   python data/prepare_dataset.py --images raw/images --labels raw/labels --names aircraft drone vehicle
   ```
   `--group-pattern` keeps all frames of one video in the same split. Without it, near-identical
   frames end up in both train and val, which inflates mAP.
   The script reports class balance and **box-size distribution**. Drones are often under 32 px, and if they are, train at 960.

## Phase 2: Training

Colab T4: open **[train/colab_train.ipynb](../train/colab_train.ipynb)**. It runs the whole flow below.

```bash
python train/train.py --model yolov8n.pt --name v8n_640                 # baseline, target mAP50 > 0.65
python train/train.py --imgsz 960 --batch 16 --name v8n_960             # small drones
python train/tune.py --iterations 30 --epochs 20 --fraction 0.5         # genetic HP search -> hyp_tuned.yaml
python train/train.py --hyp train/hyp_tuned.yaml --name v8n_640_tuned
python train/plot_curves.py runs/train/v8n_640 runs/train/v8n_960       # curves + overfitting diagnosis
python train/evaluate.py --weights weights/v8n_640.pt                   # per-class P/R/mAP on TEST split
```
- **Callbacks** ([train/callbacks.py](../train/callbacks.py)): save `best_map50.pt` on every mAP50 improvement, early-stop after
  `--patience` epochs without a gain > `--min-delta`, and log the epoch the target is first reached. Ultralytics' own
  best.pt/patience track mAP50-95, so they are disabled (`patience=0`) and this callback controls stopping.
- `train.py` scores the final model on **val and test**. Report the **test** mAP: the checkpoint was chosen on val, so val is optimistic.
- `optimizer=auto` silently ignores `lr0`/`momentum`, so `tune.py` and tuned presets use an explicit `SGD`/`AdamW`.

**If mAP50 is below 0.65**, change one thing at a time, in this order:
1. Look at the per-class table and `confusion_matrix_normalized.png` to find which class and which error (missed vs confused with background).
2. **Labels:** missed tiny drones or inconsistent boxes cap mAP more than any hyperparameter does.
3. **`--imgsz 960`:** usually the biggest gain when many drones are under 32 px.
4. **More data** for the weakest class. Add bird/sky background images if false positives dominate.
5. **More epochs** if `plot_curves.py` says the model is still improving at the last epoch.
6. **`tune.py`:** typically worth a few points, not tens.
7. **YOLOv8s:** more accurate but ~3x slower on the Nano.

## Phase 3: Optimisation (ONNX → TensorRT)

All at once (GPU machine with TensorRT, e.g. Colab T4):
```bash
bash export/run_all.sh runs/detect/yolo_custom/weights/best.pt data/dataset/images/test
```
Step by step:
```bash
# 1. ONNX (dev machine) -> weights/yolo_custom.{pt,onnx}
python -m export.export_onnx --weights runs/detect/yolo_custom/weights/best.pt --images data/dataset/images/test
# 2. TensorRT engines (build ON the device you will run them on)
python3 -m export.build_engine --onnx weights/yolo_custom.onnx --fp16
python3 -m export.build_engine --onnx weights/yolo_custom.onnx --int8 --calib-images data/dataset/images/train
# 3. Verify every export against PyTorch (raw tensors + detections)
python3 -m export.verify --reference weights/yolo_custom.pt --images data/dataset/images/test \
    --models weights/yolo_custom.onnx weights/yolo_custom_fp16.engine weights/yolo_custom_int8.engine
# 4. Latency: PyTorch FP32/FP16 vs ONNX Runtime vs TensorRT FP16/INT8 (+ breakdown chart)
python3 -m benchmark.benchmark --images data/dataset/images/test --plot --weights weights/yolo_custom.pt \
    weights/yolo_custom.pt@fp16 weights/yolo_custom.onnx weights/yolo_custom_fp16.engine weights/yolo_custom_int8.engine
```
On Colab: `pip install tensorrt pycuda onnxruntime-gpu`. Without `onnxruntime-gpu`, the ONNX row runs on CPU;
the `backend` column always shows what was actually used.
- Every backend shares the **same letterbox + decode + NMS** (`.pt` runs through `TorchDetector`), so verification
  and latency differences come from the inference engine alone.
- `verify` thresholds loosen with precision (FP32 ≥ 98% boxes matched, FP16 ≥ 95%, INT8 ≥ 85%). Confirm with
  `benchmark.eval_accuracy`, because mAP is the number that goes in the report.
- INT8 calibration uses **train** images; testing on the test split keeps the evaluation honest.
- ONNX uses **opset 12, static 1×3×640×640**, which works on TensorRT 8.2 (Nano) through 10.x.
- ⚠️ **INT8 on the Jetson Nano:** its Maxwell GPU has no fast INT8 path, so INT8 will not beat FP16 there.
  **FP16 is the right choice on the Nano.** To show INT8 gains in your report, build and benchmark INT8 on a GPU
  that supports it (Colab T4, RTX, Jetson Xavier/Orin), and say so in the text.
- Lower `--imgsz` (416/320) at export time is the biggest FPS lever on the Nano, but it costs small-object recall.

## Phase 4: Tracking (ByteTrack)

`track/byte_tracker.py` is a standalone implementation:
1. Kalman-predict all tracks → match **high-score** detections (IoU × score cost, Hungarian).
2. Match the leftover tracks with **low-score** detections (0.1-0.5), which keeps IDs through blur and occlusion.
3. New tracks start only for score ≥ `track_thresh + 0.1` and are confirmed on the second hit. Lost tracks are kept for `track_buffer` frames.

So with tracking on, the **detector threshold is set to 0.1** automatically (`deploy/pipeline.py`), because
filtering at 0.25 first would remove ByteTrack's second stage.

```bash
python -m deploy.run --weights weights/v8n_640.pt --source demo.mp4 --show --save out.mp4
```
```bash
# ONNX model: video in -> annotated video + MOT track file out, with re-identification
python -m deploy.run --weights weights/yolo_custom.onnx --source demo.mp4 --reid \
    --save out/demo_tracked.mp4 --save-txt out/demo_tracks.txt
python -m deploy.run --weights weights/yolo_custom.onnx --source 0 --show --reid      # webcam
```
**Occlusion & re-identification.** Occluded tracks are drawn as grey dashed boxes at their Kalman-predicted
position and keep their id if they re-appear within `--track-buffer` frames. With `--reid`, a 4th matching stage
(our extension, [track/byte_tracker.py](../track/byte_tracker.py)) re-links lost tracks that plain IoU matching misses:
**hovering drones** (prediction drifts away from where the drone stopped) and **small fast drones** (box moves more
than its own width per frame, so IoU = 0). It matches by centre distance to the predicted *or* last-seen position,
the same class, a similar size, and an HSV colour histogram (when the box is ≥ 10 px). This is not deep re-ID: an
object that leaves the scene for longer than the buffer comes back with a new id.

Tuning: raise `--track-buffer` for long occlusions. Raise `--track-thresh` if false tracks appear.
Lower `--match-thresh` for fast-moving drones (small boxes overlap little between frames).

## Phase 5: Edge deployment (Jetson Nano)

Full guide: **[deploy/JETSON.md](../deploy/JETSON.md)** (JetPack/CUDA/TensorRT versions, engine transfer, camera, memory/power, boot service).
```bash
bash deploy/sync_to_jetson.sh <user>@<jetson-ip>                 # laptop -> Nano (code + .onnx, never .engine)
bash deploy/jetson_setup.sh --headless                           # on the Nano, once
python3 -m export.build_engine --onnx weights/yolo_custom.onnx --fp16 --workspace 512
python3 -m deploy.check_engine weights/yolo_custom_fp16.engine
python3 -m deploy.run --weights weights/yolo_custom_fp16.engine --source csi --reid --stream-port 8080
bash deploy/service/install_service.sh                           # start automatically at boot
```

## Phase 6: Benchmarking

```bash
# on each device
python3 -m benchmark.benchmark --images data/dataset/images/test --track \
    --weights weights/v8n_640.onnx weights/v8n_640_fp16.engine weights/v8s_640_fp16.engine
# accuracy of every variant (where ultralytics is installed)
python -m benchmark.eval_accuracy --weights weights/v8n_640.pt weights/v8n_640.onnx weights/v8n_640_fp16.engine
python -m benchmark.report --device "Jetson Nano"     # -> results/report.md + tradeoff.png
```
Reports pre/infer/post/track ms, p95 latency, FPS and model size. Log power with `tegrastats` in parallel.

Suggested experiment table for the report:
| Experiment | Variable |
|---|---|
| Model size | v8n vs v8s |
| Precision | FP32 (ONNX) vs FP16 vs INT8 (TRT) |
| Input size | 320 / 416 / 640 / 960 |
| Tracking | detection-only vs ByteTrack (ID switches, count accuracy) |
| Hardware | laptop CPU (ORT) vs Nano (TRT) vs desktop GPU |

## Phase 7: Streamlit dashboard

```bash
streamlit run app.py
```
**Live tracking tab:** upload a video, use a camera (webcam / `csi` / RTSP on the machine running Streamlit) or an image.
- Live frames with track IDs, trails and occluded tracks
- Pipeline FPS (incl. UI) vs processing FPS (model + tracker only), inference ms
- Latency gauge against the real-time budget, latency timeline
- Per-class bar chart (in frame now vs unique tracked so far)
- Annotated-video download, with an inline preview when OpenCV can write H.264/VP8

**Backend comparison tab:** PyTorch vs ONNX Runtime vs TensorRT on the same frames with the same pre/post-processing.
- Speed-up per model, stage-by-stage latency chart, detection agreement with the baseline
- The same frame annotated by each backend, side by side
- Saved `benchmark/results/speed.csv` rows from other devices (e.g. copy the Jetson's file over to compare T4 vs Nano)

Streamlit needs Python ≥ 3.8, so run it on the laptop, or on a Jetson with a Python 3.8+ environment.

## Phase 8: Report and demo checklist
- [ ] Dataset card: sources, class counts, box-size histogram, split strategy
- [ ] Training curves + confusion matrix + per-class AP
- [ ] Speed/accuracy table + `tradeoff.png` per device
- [ ] Tracking demo video (`--save`) showing stable IDs through occlusion
- [ ] Failure analysis: tiny drones, birds misdetected as drones, motion blur
- [ ] Optional: MOTA/IDF1 with [TrackEval](https://github.com/JonathonLuiten/TrackEval) on 2-3 ID-annotated clips
