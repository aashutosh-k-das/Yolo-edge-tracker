# Deploying on the Jetson Nano

## 1. Setup: JetPack, CUDA, TensorRT

The original Jetson Nano (4 GB / 2 GB) tops out at **JetPack 4.6.x** (L4T R32.7.x). Flash the **latest 4.6.x
"Jetson Nano Developer Kit SD Card Image"** from NVIDIA's download centre onto a **≥ 64 GB UHS-1 microSD**
with balenaEtcher. Newer JetPack 5/6 releases do not support this board.

| Component | JetPack 4.6.x version | Consequence for this project |
|---|---|---|
| Ubuntu | 18.04 | |
| Python | **3.6.9** | Ultralytics (needs ≥ 3.8) does not run here. Our `deploy/`, `track/`, `export/build_engine.py` only need numpy/scipy/OpenCV/TensorRT |
| CUDA | 10.2 | |
| cuDNN | 8.2 | |
| TensorRT | **8.2** | ONNX must be **opset ≤ 13**; we export opset 12 |
| OpenCV | 4.1.1 **with GStreamer** | never `pip install opencv-python`: the pip build has no GStreamer, so the CSI camera stops working |
| GPU | 128-core Maxwell (SM 5.3) | fast FP16, **no fast INT8**, so use FP16 engines |

> **Jetson Orin Nano?** That is a different board: JetPack 6, Python 3.10, TensorRT 10, and INT8 *is* fast.
> All scripts here also work on it (the TensorRT loader supports the TRT 10 API).

First boot (with monitor + keyboard, or headless via the micro-USB serial console), then:
```bash
cat /etc/nv_tegra_release          # check L4T version, e.g. "R32 (release), REVISION: 7.x"
# copy the project from your laptop (run ON THE LAPTOP):
bash deploy/sync_to_jetson.sh <user>@<jetson-ip>
# on the Jetson:
cd ~/yolo-edge-tracker && bash deploy/jetson_setup.sh            # add --headless to drop the desktop
```
`jetson_setup.sh` does the following:
- prints the versions above
- sets MAXN power mode and locks the clocks
- adds 4 GB of swap
- installs numpy 1.19.4 (1.19.5 crashes with *Illegal instruction* on the Nano), scipy and pycuda
- adds you to the `video` group
- installs `jtop`
- runs an import self-test

## 2. Transferring and loading the engine

**Do not copy a `.engine` built on Colab or your laptop.** An engine only runs on the GPU architecture *and*
TensorRT version it was built with (T4 = SM 7.5 + TRT 10 ≠ Nano = SM 5.3 + TRT 8.2). Copy the **`.onnx`** and build on the Nano:

```bash
# laptop: copies code, weights/*.onnx, data/dataset.yaml (class names). Never copies .engine files
bash deploy/sync_to_jetson.sh <user>@<jetson-ip>

# Jetson: build (5-20 min for YOLOv8n). --workspace 512 avoids running out of memory on the 4 GB board
python3 -m export.build_engine --onnx weights/yolo_custom.onnx --fp16 --workspace 512

# load + sanity-check: versions, I/O shapes, class names, GPU memory, inference latency
python3 -m deploy.check_engine weights/yolo_custom_fp16.engine
```
In your own code, loading is two lines (`deploy/detectors.py` handles TRT 8 and 10 APIs, pinned memory, CUDA context):
```python
from deploy.detectors import load_detector
det = load_detector("weights/yolo_custom_fp16.engine")   # dets = det(frame_bgr) -> (N, 6) x1,y1,x2,y2,conf,cls
```
Raw engines carry no class names, so they come from `data/dataset.yaml` (`aircraft, drone, vehicle`).

## 3. Real-time camera inference

```bash
# CSI camera (Raspberry Pi Cam v2 / IMX219), window on the attached monitor
python3 -m deploy.run --weights weights/yolo_custom_fp16.engine --source csi --reid --show
# USB webcam, headless: watch the annotated stream from your laptop at http://<jetson-ip>:8080
python3 -m deploy.run --weights weights/yolo_custom_fp16.engine --source 0 --reid --stream-port 8080
# IP camera with hardware H.264 decoding
python3 -m deploy.run --weights weights/yolo_custom_fp16.engine --source rtsp://<cam>/stream --gst
```
What makes it real-time on a Nano ([deploy/camera.py](camera.py)):
- **Latest-frame thread:** a background thread keeps only the newest camera frame. Without it, OpenCV's buffer fills
  whenever inference is slower than the camera, and you end up processing frames that are seconds old. Skipped
  frames are reported at the end of the run.
- **CSI via GStreamer** (`nvarguscamerasrc` + `nvvidconv`): debayering, scaling and colour conversion run in
  hardware, not on the CPU.
- **USB cameras use MJPG** (720p30 over USB 2; raw YUYV usually only manages ~10 fps at 720p). `--gst` uses the
  hardware H.264 decoder for RTSP.
- **MJPEG stream** (`--stream-port`): frames are only encoded while a browser is connected. It has no
  authentication, so only use it on a trusted network.

## 4. Memory and power optimisation

**Power and thermals**
| Action | How | Why |
|---|---|---|
| MAXN mode | `sudo nvpmodel -m 0` (persists) | 10 W budget, all 4 CPU cores. Mode 1 = 5 W, 2 cores, lower clocks |
| Lock clocks | `sudo jetson_clocks` (not persistent; the boot service runs it) | stable FPS, no DVFS ramp-up delays |
| Power supply | **5 V / 4 A barrel jack + J48 jumper** | micro-USB (5 V / 2 A) brown-outs under GPU load → random reboots |
| Cooling | 40 mm fan on the heatsink | the stock passive heatsink throttles under sustained load |
| Measure it | `sudo tegrastats --interval 500 --logfile power.log` → `python3 -m benchmark.parse_tegrastats power.log --fps <fps>` | average/peak W, **FPS per watt**, mJ per frame, temperatures for your report |

**Memory** (4 GB shared between CPU and GPU)
| Action | Saving |
|---|---|
| Headless boot: `bash deploy/jetson_setup.sh --headless` | several hundred MB, the largest single saving |
| Don't install torch / ultralytics / onnxruntime on the Nano | the deploy path needs only TensorRT + numpy |
| Build engines with `--workspace 512` and swap enabled | prevents builder OOM. Swap is for building, not running: if `parse_tegrastats` reports swap use at runtime, FPS suffers |
| FP16 engine | half the weight memory of FP32 |
| Pre-allocated pinned buffers (already in `TensorRTDetector`) | no per-frame allocations |
| Don't `--save` video in long runs | writing 720p mp4 costs CPU *and* SD-card space. Use `--save-txt` (tracks) + `--stream-port` |

**Speed** (in order of impact)
1. **FP16 TensorRT** instead of ONNX Runtime/PyTorch.
2. **Smaller input:** `export_onnx --imgsz 416` (≈2.4x fewer pixels than 640). Measure the mAP cost on small drones before committing.
3. **Smaller camera frames:** `--width 960 --height 540` reduces resize and draw cost on the CPU.
4. **Headless:** `--show` (X11 window) costs a few ms per frame. `--stream-port` costs nothing when nobody watches.
5. The CPU side (letterbox, NMS, tracking) is often 30-40% of the frame time on the Nano's A57 cores. Check the
   `pre_ms / post_ms / track_ms` columns of `benchmark.benchmark` before optimising the model further.

## 5. Running at boot (systemd)

```bash
nano deploy/service/tracker.env              # WEIGHTS, SOURCE, EXTRA_ARGS
bash deploy/service/install_service.sh       # installs, enables and starts 'yolo-tracker'
journalctl -u yolo-tracker -f                # live logs
```
The service ([yolo-tracker.service](service/yolo-tracker.service)):
- runs as your user (in the `video` group), after the network and the CSI camera daemon are up
- runs `jetson_clocks` as root first, since clock locking doesn't survive a reboot
- restarts automatically 5 s after a crash
- stops with SIGTERM, which `run.py` handles, so open video/track files are closed properly

Each boot writes `logs/tracks_<date>_<time>.txt` (MOT format). Watch it live at `http://<jetson-ip>:8080`.
Change settings: edit `tracker.env` → `sudo systemctl restart yolo-tracker`.
Free the camera for manual testing: `sudo systemctl stop yolo-tracker`.
Remove: `bash deploy/service/install_service.sh --uninstall`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Illegal instruction (core dumped)` on `import numpy` | `export OPENBLAS_CORETYPE=ARMV8` or use numpy 1.19.4 |
| Engine fails to deserialize | it was built elsewhere / with another TRT version: rebuild on the Nano |
| `Could not open video source: csi` | `sudo systemctl restart nvargus-daemon`; check the ribbon cable orientation |
| Engine build killed / freezes | swap missing → `jetson_setup.sh`; use `--workspace 256`; close the desktop |
| FPS drops after a few minutes | thermal throttling: check `GPU@..C` in tegrastats, add a fan |
| Random reboots under load | power supply: use the barrel jack + J48 jumper |
