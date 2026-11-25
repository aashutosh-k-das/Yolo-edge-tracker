#!/usr/bin/env bash
# One-time Jetson Nano setup (JetPack 4.6.x). Run ON the Nano from the project root:
#   bash deploy/jetson_setup.sh [--headless]
# --headless disables the desktop at boot (frees several hundred MB of the shared 4 GB RAM).
set -euo pipefail

echo "== 1. Versions =="
head -n1 /etc/nv_tegra_release || { echo "Not a Jetson?"; exit 1; }
/usr/local/cuda/bin/nvcc --version | tail -n1 || echo "CUDA not on PATH (fine, it is in /usr/local/cuda)"
python3 -c "import tensorrt as trt; print('TensorRT', trt.__version__)"
python3 -c "import cv2; b=cv2.getBuildInformation(); print('OpenCV', cv2.__version__, '| GStreamer:', 'YES' in b.split('GStreamer:')[1].split('\n')[0])"
python3 --version

echo "== 2. Max performance: nvpmodel 0 = MAXN (10 W). Needs the 5V/4A barrel jack + J48 jumper =="
sudo nvpmodel -m 0
sudo nvpmodel -q | tail -n1
sudo jetson_clocks

echo "== 3. 4 GB swap file (engine builds can OOM without it) =="
if ! swapon --show | grep -q /swapfile; then
    sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
    sudo mkswap /swapfile && sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
free -h | sed -n '1,3p'

echo "== 4. Python packages (system python3 - keep JetPack's OpenCV + TensorRT) =="
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-yaml libopenblas-base
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include:${CPATH:-}
export LIBRARY_PATH=/usr/local/cuda/lib64:${LIBRARY_PATH:-}
python3 -m pip install --user --upgrade "pip<22"
python3 -m pip install --user numpy==1.19.4            # 1.19.5 has the "Illegal instruction" bug on Nano
python3 -m pip install --user --only-binary=:all: scipy==1.5.4 \
    || sudo apt-get install -y python3-scipy           # fall back to the apt build instead of a 1-hour compile
python3 -m pip install --user pycuda==2021.1           # compiles against CUDA 10.2 (~5-10 min)
grep -q OPENBLAS_CORETYPE ~/.bashrc || echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc

echo "== 5. Camera access =="
sudo usermod -aG video "$USER"
ls /dev/video* 2>/dev/null || echo "no /dev/video* yet (CSI cameras show up as /dev/video0 once connected)"

echo "== 6. Monitoring tool (jtop) =="
sudo -H python3 -m pip install -U jetson-stats || true

if [ "${1:-}" = "--headless" ]; then
    echo "== 7. Boot to console (no desktop). Undo: sudo systemctl set-default graphical.target =="
    sudo systemctl set-default multi-user.target
fi

echo "== Self-test =="
OPENBLAS_CORETYPE=ARMV8 python3 -c "import numpy, scipy, pycuda.driver, tensorrt, cv2, yaml; from track import BYTETracker; print('imports OK')"
echo "Done. Log out/in (video group), then:"
echo "  python3 -m export.build_engine --onnx weights/<model>.onnx --fp16 --workspace 512"
echo "  python3 -m deploy.check_engine weights/<model>_fp16.engine"
