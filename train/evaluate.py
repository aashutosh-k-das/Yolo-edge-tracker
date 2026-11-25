"""Phase 2 - evaluate a trained model on the held-out TEST split with a per-class breakdown.

    python train/evaluate.py --weights weights/v8n_640.pt
"""
import argparse
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "dataset.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    model = YOLO(a.weights)
    m = model.val(data=a.data, split=a.split, imgsz=a.imgsz, device=a.device, batch=16,
                  conf=0.001, iou=0.6, plots=True, verbose=False,
                  project=os.path.join(ROOT, "runs", "eval"),
                  name=os.path.splitext(os.path.basename(a.weights))[0], exist_ok=True)

    print("\n{:<12}{:>9}{:>9}{:>9}{:>11}".format("class", "P", "R", "mAP50", "mAP50-95"))
    print("{:<12}{:>9.3f}{:>9.3f}{:>9.3f}{:>11.3f}".format("all", m.box.mp, m.box.mr, m.box.map50, m.box.map))
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        print("{:<12}{:>9.3f}{:>9.3f}{:>9.3f}{:>11.3f}".format(model.names[int(c)], p, r, ap50, ap))
    print("\nSpeed (ms/img):", {k: round(v, 2) for k, v in m.speed.items()})
    print("Plots (PR curve, confusion matrix) saved to", m.save_dir)


if __name__ == "__main__":
    main()
