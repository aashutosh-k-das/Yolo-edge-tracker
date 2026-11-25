"""Phase 6b - accuracy (mAP) of every exported variant on the TEST split.

    python -m benchmark.eval_accuracy --weights weights/v8n_640.pt weights/v8n_640.onnx \
        weights/v8n_640_fp16.engine weights/v8n_640_int8.engine

Uses Ultralytics' validator, which can load .pt, .onnx and .engine files, so every
variant is scored with the identical metric code. Run where Ultralytics is installed
(GPU desktop / Colab, or a Jetson with Python >= 3.8). Engines must be evaluated on the
device type they were built for.
"""
import argparse
import csv
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "dataset.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark", "results", "accuracy.csv"))
    a = ap.parse_args()

    rows = []
    for w in a.weights:
        print("Evaluating", w)
        m = YOLO(w, task="detect").val(data=a.data, split=a.split, imgsz=a.imgsz, batch=1,
                                       conf=0.001, iou=0.6, plots=False, verbose=False)
        row = {"model": os.path.basename(w), "mAP50": round(m.box.map50, 4),
               "mAP50_95": round(m.box.map, 4), "precision_P": round(m.box.mp, 4),
               "recall_R": round(m.box.mr, 4)}
        for i, c in enumerate(m.box.ap_class_index):
            row["AP50_95_" + m.names[int(c)]] = round(m.box.class_result(i)[3], 4)
        rows.append(row)
        print("  mAP50={mAP50} mAP50-95={mAP50_95}".format(**row))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
