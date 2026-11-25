"""Phase 3a - export a trained .pt to ONNX (opset 12, simplified, static 1x3xHxW) and verify it.

    python -m export.export_onnx --weights runs/detect/yolo_custom/weights/best.pt \
        --images data/dataset/images/test
    -> weights/yolo_custom.pt (copy of the checkpoint) + weights/yolo_custom.onnx

Why these settings:
  opset 12      parses on TensorRT 8.2 (Jetson Nano, JetPack 4.6) as well as TensorRT 8.6 / 10.x
  simplify      constant-folds the graph (onnxslim) -> fewer nodes, cleaner TensorRT build
  static shape  batch 1, fixed HxW: TensorRT can pick the fastest kernels; no optimisation profiles
  fp32 ONNX     precision is chosen later, at TensorRT build time (FP16 / INT8)
"""
import argparse
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_name(weights):
    """runs/detect/yolo_custom/weights/best.pt -> 'yolo_custom'; other paths -> file stem."""
    stem = os.path.splitext(os.path.basename(weights))[0]
    parent = os.path.dirname(os.path.abspath(weights))
    if stem in ("best", "last") and os.path.basename(parent) == "weights":
        return os.path.basename(os.path.dirname(parent))
    return stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", default=None, help="output base name (default: run folder name)")
    ap.add_argument("--imgsz", type=int, default=640, help="320/416 = faster on Jetson, worse on tiny drones")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--images", default=None, help="folder of real images used to verify the export")
    ap.add_argument("-n", type=int, default=25)
    a = ap.parse_args()

    name = a.name or default_name(a.weights)
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
    pt_dst = os.path.join(ROOT, "weights", name + ".pt")
    if os.path.abspath(a.weights) != os.path.abspath(pt_dst):
        shutil.copy2(a.weights, pt_dst)

    from ultralytics import YOLO
    path = YOLO(pt_dst).export(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=True,
                               dynamic=False, half=False, batch=1)  # NMS stays outside the graph
    dst = os.path.splitext(pt_dst)[0] + ".onnx"
    if os.path.abspath(str(path)) != os.path.abspath(dst):
        shutil.move(str(path), dst)
    print("\nONNX model: {} ({:.1f} MB)".format(dst, os.path.getsize(dst) / 1e6))

    # ---------------------------------------------------------------- structural checks
    import onnx
    model = onnx.load(dst)
    onnx.checker.check_model(model)
    shape = lambda v: [d.dim_value if d.dim_value else d.dim_param for d in v.type.tensor_type.shape.dim]
    ins, outs = [shape(i) for i in model.graph.input], [shape(o) for o in model.graph.output]
    meta = {p.key: p.value for p in model.metadata_props}
    print("opset   :", [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")])
    print("input   :", ins[0], "| output:", outs[0], "| nodes:", len(model.graph.node))
    print("names   :", meta.get("names"))

    nc = len(YOLO(pt_dst).names)
    ok_in = ins[0] == [1, 3, a.imgsz, a.imgsz]
    ok_out = len(outs) == 1 and len(outs[0]) == 3 and outs[0][1] == 4 + nc and isinstance(outs[0][2], int)
    if not (ok_in and ok_out):
        raise SystemExit("Unexpected I/O. Expected input [1,3,{0},{0}] and output [1,{1},A]. This pipeline "
                         "supports YOLOv8-style heads (not NMS-free YOLOv10/YOLO26 heads).".format(a.imgsz, 4 + nc))
    print("static shapes OK: [1, 4+{} classes, {} anchors]".format(nc, outs[0][2]))

    # ---------------------------------------------------------------- numerical check
    if not a.images:
        print("\nNo --images given; skipping numerical verification (recommended!).")
        return
    from deploy.detectors import load_detector
    from export.verify import judge, load_images, verify
    images = load_images(a.images, a.n)
    res = verify(load_detector(pt_dst, imgsz=a.imgsz, device="cpu"), load_detector(dst), images)
    print("\nPyTorch vs ONNX on {} images: {}".format(len(images), res))
    print("Verdict:", judge(res, "fp32"))


if __name__ == "__main__":
    main()
