"""Phase 3b - build a TensorRT engine from ONNX (FP32 / FP16 / INT8).

!! Run this ON THE TARGET DEVICE (e.g. the Jetson Nano). TensorRT engines are tied to the
   GPU architecture and TensorRT version they were built with.

    python -m export.build_engine --onnx weights/v8n_640.onnx --fp16
    python -m export.build_engine --onnx weights/v8n_640.onnx --int8 --calib-images data/dataset/images/train

Equivalent trtexec command (handy for cross-checking):
    /usr/src/tensorrt/bin/trtexec --onnx=weights/v8n_640.onnx --saveEngine=weights/v8n_640_fp16.engine --fp16
"""
import argparse
import os
import time

import pycuda.autoinit  # noqa: F401  (creates CUDA context for the calibrator)
import tensorrt as trt


def build(onnx_path, engine_path, fp16=False, int8=False, calib_images=None, calib_count=500,
          workspace_mb=1024, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX file")

    for i in range(network.num_inputs):
        t = network.get_input(i)
        print("input  {} {} {}".format(t.name, tuple(t.shape), t.dtype))
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print("output {} {} {}".format(t.name, tuple(t.shape), t.dtype))

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):  # TRT >= 8.4
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    else:                                          # TRT 8.2 (Jetson Nano)
        config.max_workspace_size = workspace_mb << 20

    if fp16 or int8:
        if not builder.platform_has_fast_fp16:
            print("WARNING: platform has no fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)  # also FP16 fallback for layers INT8 can't handle
    if int8:
        if not builder.platform_has_fast_int8:
            print("WARNING: this GPU has no fast INT8 path (e.g. Jetson Nano / Maxwell). "
                  "The INT8 engine will build but is unlikely to beat FP16 here.")
        from export.calibrator import YoloEntropyCalibrator
        h = network.get_input(0).shape[2]
        cache = os.path.splitext(engine_path)[0] + ".calib.cache"
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = YoloEntropyCalibrator(calib_images, imgsz=h,
                                                       max_images=calib_count, cache_file=cache)

    print("Building engine (this can take 5-20 min on a Jetson Nano)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print("Saved {} ({:.1f} MB) in {:.0f}s".format(engine_path, os.path.getsize(engine_path) / 1e6,
                                                   time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-images", default=None, help="folder of representative images (INT8)")
    ap.add_argument("--calib-count", type=int, default=500)
    ap.add_argument("--workspace", type=int, default=1024, help="MB (use 512 on a 4GB Nano if OOM)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.int8 and not a.calib_images:
        ap.error("--int8 requires --calib-images")
    prec = "int8" if a.int8 else ("fp16" if a.fp16 else "fp32")
    out = a.output or "{}_{}.engine".format(os.path.splitext(a.onnx)[0], prec)
    print("TensorRT", trt.__version__, "| precision:", prec)
    build(a.onnx, out, a.fp16, a.int8, a.calib_images, a.calib_count, a.workspace, a.verbose)


if __name__ == "__main__":
    main()
