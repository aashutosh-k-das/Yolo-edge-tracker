"""Phase 3c - verify that exported models behave like the original PyTorch model.

    python -m export.verify --reference weights/yolo_custom.pt \
        --models weights/yolo_custom.onnx weights/yolo_custom_fp16.engine weights/yolo_custom_int8.engine \
        --images data/dataset/images/test

Two levels of checks, on identical inputs (same letterbox, same NMS for every backend):
  1. RAW tensor   - the (1, 4+nc, 8400) output: max |diff| of box coords and of class scores.
  2. DETECTIONS   - after NMS at conf 0.25: how many reference boxes are found again
                    (same class, IoU >= 0.5), their mean IoU and mean |score diff|.
Thresholds are looser for lower precision, since FP16/INT8 are *expected* to drift slightly.
Detection agreement is a sanity check; the real accuracy number is mAP:
    python -m benchmark.eval_accuracy --weights <all variants>
"""
import argparse
import glob
import json
import os

import numpy as np

# precision -> (min match rate, min mean IoU, max mean |score diff|)
THRESHOLDS = {"fp32": (0.98, 0.98, 0.01), "fp16": (0.95, 0.97, 0.02), "int8": (0.85, 0.93, 0.06)}


def precision_of(path):
    n = os.path.basename(path).lower()
    return "int8" if "int8" in n else ("fp16" if "fp16" in n or n.endswith("@fp16") else "fp32")


def box_iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_detections(ref, cand, iou_thr=0.5):
    """Greedy one-to-one matching of (N,6) detections, same class only, highest IoU first.
    Returns (n_ref, n_cand, matched_ious, matched_score_diffs)."""
    if len(ref) == 0 or len(cand) == 0:
        return len(ref), len(cand), [], []
    iou = box_iou(ref[:, :4], cand[:, :4])
    iou[ref[:, 5][:, None] != cand[:, 5][None, :]] = 0.0
    ious, sdiff = [], []
    while True:
        i, j = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[i, j] < iou_thr:
            break
        ious.append(float(iou[i, j]))
        sdiff.append(abs(float(ref[i, 4] - cand[j, 4])))
        iou[i, :] = 0.0
        iou[:, j] = 0.0
    return len(ref), len(cand), ious, sdiff


def load_images(folder, n, seed=0):
    import cv2
    files = sorted(f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                   if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    np.random.RandomState(seed).shuffle(files)
    imgs = [cv2.imread(f) for f in files[:n]]
    return [im for im in imgs if im is not None]


def verify(ref_det, cand_det, images):
    from deploy.postprocess import postprocess
    from deploy.utils import preprocess

    raw_box, raw_cls = [], []
    n_ref = n_cand = 0
    ious, sdiffs = [], []
    for img in images:
        # 1) raw tensors on the identical fp32 blob
        blob, ratio, pad = preprocess(img, ref_det.imgsz)
        r = ref_det.infer(blob)
        c = cand_det.infer(blob.astype(cand_det.input_dtype, copy=False)).astype(np.float32)
        if r.shape != c.shape:
            raise ValueError("output shape mismatch: reference {} vs candidate {}".format(r.shape, c.shape))
        raw_box.append(float(np.abs(r[:, :4] - c[:, :4]).max()))
        raw_cls.append(float(np.abs(r[:, 4:] - c[:, 4:]).max()))
        # 2) decoded detections
        dr = postprocess(r, ratio, pad, img.shape[:2], 0.25, 0.45)
        dc = postprocess(c, ratio, pad, img.shape[:2], 0.25, 0.45)
        a, b, i, s = match_detections(dr, dc)
        n_ref, n_cand = n_ref + a, n_cand + b
        ious += i
        sdiffs += s
    return {
        "raw_max_abs_box_px": round(max(raw_box), 4),
        "raw_max_abs_score": round(max(raw_cls), 5),
        "ref_dets": n_ref, "cand_dets": n_cand,
        "match_rate": round(len(ious) / n_ref, 4) if n_ref else None,
        "extra_dets": n_cand - len(ious),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "mean_score_diff": round(float(np.mean(sdiffs)), 5) if sdiffs else None,
    }


def judge(res, precision):
    if res["match_rate"] is None:
        return "NO-DETS"  # reference found nothing: use real test images
    mr, miou, msd = THRESHOLDS[precision]
    ok = res["match_rate"] >= mr and res["mean_iou"] >= miou and res["mean_score_diff"] <= msd
    return "PASS" if ok else "CHECK"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="the trained .pt (fp32)")
    ap.add_argument("--models", nargs="+", required=True, help=".onnx / .engine files to verify")
    ap.add_argument("--images", required=True, help="real images, e.g. data/dataset/images/test")
    ap.add_argument("-n", type=int, default=50, help="number of images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="benchmark/results/verify.json")
    a = ap.parse_args()

    from deploy.detectors import load_detector
    images = load_images(a.images, a.n)
    if not images:
        raise SystemExit("No images found in " + a.images)
    print("Reference: {} on {} images".format(a.reference, len(images)))
    ref = load_detector(a.reference, imgsz=a.imgsz)

    rows = []
    for m in a.models:
        print("Verifying", m, "...")
        det = load_detector(m, imgsz=a.imgsz)
        if det.imgsz != ref.imgsz:
            raise SystemExit("{} expects {}px input but reference uses {}px".format(m, det.imgsz, ref.imgsz))
        res = verify(ref, det, images)
        det.close()
        res.update({"model": os.path.basename(m), "backend": det.backend, "precision": precision_of(m)})
        res["verdict"] = judge(res, res["precision"])
        rows.append(res)

    cols = ["model", "precision", "raw_max_abs_box_px", "raw_max_abs_score", "match_rate",
            "mean_iou", "mean_score_diff", "extra_dets", "verdict"]
    print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print("\nPASS thresholds (match rate, mean IoU, score diff):", THRESHOLDS)
    print("Raw box diffs of a few px are normal for FP16/INT8; what matters is the detection-level match.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
