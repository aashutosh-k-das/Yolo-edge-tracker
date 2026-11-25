"""Unified detector interface over three backends.

    det = load_detector("weights/best.pt")        # Ultralytics / PyTorch (dev machine)
    det = load_detector("weights/best.onnx")      # ONNX Runtime (CPU / CUDA / CoreML)
    det = load_detector("weights/best.engine")    # TensorRT (Jetson / NVIDIA GPU)
    dets = det(frame_bgr)                         # (N, 6) [x1, y1, x2, y2, conf, cls]
    det.timings                                   # {'pre': ms, 'infer': ms, 'post': ms}

The ONNX and TensorRT backends use only numpy + cv2 + the runtime itself (no torch,
no ultralytics), so they run on a Jetson Nano with the stock JetPack 4.6 Python 3.6.
"""
import json
import os
import time

import numpy as np

from .postprocess import postprocess
from .utils import load_names, preprocess


class BaseDetector(object):
    backend = "base"

    def __init__(self, imgsz=640, conf=0.25, iou=0.45, names=None, max_det=300):
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.names = load_names(names)
        self.input_dtype = np.float32
        self.timings = {"pre": 0.0, "infer": 0.0, "post": 0.0}

    def infer(self, blob):
        raise NotImplementedError

    def __call__(self, frame):
        t0 = time.perf_counter()
        blob, ratio, pad = preprocess(frame, self.imgsz, self.input_dtype)
        t1 = time.perf_counter()
        out = self.infer(blob)
        t2 = time.perf_counter()
        dets = postprocess(out, ratio, pad, frame.shape[:2], self.conf, self.iou, self.max_det)
        t3 = time.perf_counter()
        self.timings = {"pre": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3}
        return dets

    def warmup(self, n=10):
        dummy = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(n):
            self(dummy)

    def close(self):
        pass


# --------------------------------------------------------------------- PyTorch
class UltralyticsDetector(BaseDetector):
    backend = "pytorch"

    def __init__(self, weights, device=None, half=False, **kw):
        from ultralytics import YOLO
        super(UltralyticsDetector, self).__init__(**kw)
        self.model = YOLO(weights)
        self.device = device
        self.half = half
        if kw.get("names") is None and getattr(self.model, "names", None):
            self.names = load_names(self.model.names)

    def __call__(self, frame):
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               max_det=self.max_det, device=self.device, half=self.half,
                               verbose=False)[0]
        s = r.speed
        self.timings = {"pre": s["preprocess"], "infer": s["inference"], "post": s["postprocess"]}
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                               b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32)


class TorchDetector(BaseDetector):
    """Raw PyTorch forward pass with the SAME letterbox / decode / NMS as the ONNX and TensorRT
    backends, so a benchmark or verification isolates the inference engine only.
    (UltralyticsDetector uses Ultralytics' own pre-processing, which pads differently.)"""
    backend = "pytorch"

    def __init__(self, weights, device=None, half=False, **kw):
        import torch
        from ultralytics import YOLO
        super(TorchDetector, self).__init__(**kw)
        self.torch = torch
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.half = bool(half) and self.device.type == "cuda"
        yolo = YOLO(weights)
        if kw.get("names") is None:
            self.names = load_names(yolo.names)
        self.model = yolo.model.fuse().eval().to(self.device)
        self.model = self.model.half() if self.half else self.model.float()
        self.backend = "pytorch-{}{}".format(self.device.type, "-fp16" if self.half else "")

    def infer(self, blob):
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(blob).to(self.device)
            x = x.half() if self.half else x
            y = self.model(x)
            y = y[0] if isinstance(y, (list, tuple)) else y  # eval mode returns (preds, raw_feats)
            return y.float().cpu().numpy()                    # .cpu() also synchronises CUDA


# ---------------------------------------------------------------- ONNX Runtime
class OnnxDetector(BaseDetector):
    backend = "onnxruntime"
    PREFERRED = ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                 "CoreMLExecutionProvider", "CPUExecutionProvider"]

    def __init__(self, weights, providers=None, **kw):
        import onnxruntime as ort
        super(OnnxDetector, self).__init__(**kw)
        available = ort.get_available_providers()
        providers = providers or [p for p in self.PREFERRED if p in available]
        # TensorRT EP is opt-in: it rebuilds an engine on first run (slow start).
        if "TensorrtExecutionProvider" in providers and os.environ.get("ORT_TRT") != "1":
            providers = [p for p in providers if p != "TensorrtExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(weights, sess_options=so, providers=providers)
        self.providers = self.session.get_providers()
        self.backend = "onnxruntime-" + self.providers[0].replace("ExecutionProvider", "").lower()

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if isinstance(inp.shape[2], int):
            self.imgsz = inp.shape[2]
        if inp.type == "tensor(float16)":
            self.input_dtype = np.float16
        meta = self.session.get_modelmeta().custom_metadata_map
        if kw.get("names") is None and "names" in meta:
            self.names = load_names(meta["names"])

    def infer(self, blob):
        return self.session.run(None, {self.input_name: blob})[0]


# -------------------------------------------------------------------- TensorRT
def read_engine_file(path):
    """Return (engine_bytes, metadata). Ultralytics-exported .engine files prepend
    a 4-byte length + JSON metadata block; raw trtexec / build_engine.py files do not."""
    with open(path, "rb") as f:
        data = f.read()
    try:
        n = int.from_bytes(data[:4], byteorder="little")
        if 0 < n < min(len(data), 1 << 20):
            meta = json.loads(data[4:4 + n].decode("utf-8"))
            if isinstance(meta, dict):
                return data[4 + n:], meta
    except ValueError:  # includes UnicodeDecodeError / JSONDecodeError
        pass
    return data, None


class TensorRTDetector(BaseDetector):
    """Works with TensorRT 8.x (JetPack 4.6 on Nano) and TensorRT 10.x (new tensor API).

    Uses an explicitly created pycuda context (pushed/popped per call) instead of
    pycuda.autoinit, so it also works from Streamlit's worker threads.
    """
    backend = "tensorrt"

    def __init__(self, weights, device=0, **kw):
        import pycuda.driver as cuda
        import tensorrt as trt
        super(TensorRTDetector, self).__init__(**kw)
        self.cuda, self.trt = cuda, trt
        cuda.init()
        self.ctx = cuda.Device(device).make_context()
        try:
            engine_bytes, meta = read_engine_file(weights)
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize {} - was it built on this device with "
                                   "the same TensorRT version?".format(weights))
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.trt10 = hasattr(self.engine, "num_io_tensors")
            self._allocate()
        finally:
            self.ctx.pop()

        if meta and kw.get("names") is None and "names" in meta:
            self.names = load_names(meta["names"])

    def _allocate(self):
        cuda, trt, eng = self.cuda, self.trt, self.engine
        self.inputs, self.outputs, self.bindings = [], [], []
        if self.trt10:
            io = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
            specs = [(n, tuple(eng.get_tensor_shape(n)), trt.nptype(eng.get_tensor_dtype(n)),
                      eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT) for n in io]
        else:
            specs = [(eng.get_binding_name(i), tuple(eng.get_binding_shape(i)),
                      trt.nptype(eng.get_binding_dtype(i)), eng.binding_is_input(i))
                     for i in range(eng.num_bindings)]
        for name, shape, dtype, is_input in specs:
            if any(d < 0 for d in shape):
                raise ValueError("Dynamic shapes not supported here; export with dynamic=False.")
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            buf = {"name": name, "shape": shape, "host": host, "dev": dev}
            self.bindings.append(int(dev))
            if self.trt10:
                self.context.set_tensor_address(name, int(dev))
            (self.inputs if is_input else self.outputs).append(buf)
            if is_input:
                self.imgsz = shape[2]
                self.input_dtype = dtype

    def infer(self, blob):
        cuda = self.cuda
        self.ctx.push()
        try:
            inp = self.inputs[0]
            np.copyto(inp["host"], blob.ravel())
            cuda.memcpy_htod_async(inp["dev"], inp["host"], self.stream)
            if self.trt10:
                self.context.execute_async_v3(self.stream.handle)
            else:
                self.context.execute_async_v2(self.bindings, self.stream.handle)
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out["host"], out["dev"], self.stream)
            self.stream.synchronize()
            # YOLOv8 detect has a single output: (1, 4+nc, 8400)
            out = self.outputs[0]
            return out["host"].reshape(out["shape"]).copy()
        finally:
            self.ctx.pop()

    def close(self):
        if getattr(self, "ctx", None) is not None:
            self.ctx.push()
            del self.context, self.engine
            self.ctx.pop()
            self.ctx.detach()
            self.ctx = None


# ------------------------------------------------------------------- factory
def load_detector(weights, pt_backend="torch", **kw):
    """weights may end in '@fp16' for PyTorch half precision on CUDA, e.g. 'best.pt@fp16'.
    pt_backend: 'torch' (shared pre/post, default) or 'ultralytics' (model.predict)."""
    if weights.endswith("@fp16"):
        weights, kw["half"] = weights[:-5], True
    ext = os.path.splitext(weights)[1].lower()
    if ext == ".pt":
        cls = UltralyticsDetector if pt_backend == "ultralytics" else TorchDetector
        return cls(weights, **kw)
    if ext == ".onnx":
        return OnnxDetector(weights, **kw)
    if ext in (".engine", ".trt", ".plan"):
        return TensorRTDetector(weights, **kw)
    raise ValueError("Unsupported weights format: {}".format(weights))
