"""Load a TensorRT engine on the Jetson and check it: versions, I/O, GPU memory, latency.

    python3 -m deploy.check_engine weights/yolo_custom_fp16.engine

Run this first after building/copying an engine. Typical failures and what they mean:
  "deserialize ... failed" / "version"   engine built with another TensorRT version or GPU
                                          (e.g. on Colab) -> rebuild ON this device from the .onnx
  output shape is not (1, 4+nc, 8400)     not a YOLOv8 detect head / wrong export settings
"""
import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()

    import pycuda.driver as cuda
    import tensorrt as trt
    from deploy.detectors import TensorRTDetector

    cuda.init()
    dev = cuda.Device(0)
    print("TensorRT {} | GPU {} | compute capability {}.{}".format(
        trt.__version__, dev.name(), *dev.compute_capability()))
    try:
        with open("/etc/nv_tegra_release") as f:
            print("L4T:", f.readline().strip())
    except IOError:
        pass

    det = TensorRTDetector(a.engine)
    det.ctx.push()
    free0, total = cuda.mem_get_info()
    det.ctx.pop()
    for kind, bufs in (("input ", det.inputs), ("output", det.outputs)):
        for b in bufs:
            print("{} {:<10} shape={} dtype={}".format(kind, b["name"], b["shape"], b["host"].dtype))
    nc = det.outputs[0]["shape"][1] - 4
    print("classes: {} -> {}".format(nc, det.names))
    if len(det.names) != nc:
        print("WARNING: {} class names but the engine has {} classes. Check data/dataset.yaml".format(
            len(det.names), nc))

    blob = np.random.rand(1, 3, det.imgsz, det.imgsz).astype(det.input_dtype)
    for _ in range(10):
        det.infer(blob)
    ts = []
    for _ in range(a.iters):
        t = time.perf_counter()
        det.infer(blob)
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts)
    print("inference only: mean {:.2f} ms | p50 {:.2f} | p95 {:.2f} -> {:.1f} FPS (excl. pre/post/tracking)".format(
        ts.mean(), np.percentile(ts, 50), np.percentile(ts, 95), 1000.0 / ts.mean()))
    print("GPU memory: {:.0f} MB free of {:.0f} MB (shared with the CPU on Jetson)".format(free0 / 2**20, total / 2**20))
    det.close()
    print("OK - engine loads and runs.")


if __name__ == "__main__":
    main()
