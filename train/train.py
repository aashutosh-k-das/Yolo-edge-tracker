"""Phase 2 - train YOLOv8 on the custom dataset (GPU: Colab T4 / Kaggle / lab machine).

    # baseline (T4 defaults: batch 32, AMP, RAM cache)
    python train/train.py --model yolov8n.pt --name v8n_640
    # small drones -> larger input
    python train/train.py --model yolov8n.pt --imgsz 960 --batch 16 --name v8n_960
    # with hyper-parameters found by train/tune.py
    python train/train.py --hyp train/hyp_tuned.yaml --name v8n_640_tuned
    # Colab disconnected? continue from last.pt
    python train/train.py --name v8n_640 --resume

Callbacks (train/callbacks.py): best-by-mAP50 checkpoint + early stopping on mAP50 + target check.
After training, the best model is scored on the val AND the held-out test split. Report the TEST
number: the checkpoint was *selected* on val, so val mAP is optimistically biased.
"""
import argparse
import json
import os
import shutil
import sys

import yaml
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train.callbacks import MetricCheckpointEarlyStopping  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained start point")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "dataset.yaml"))
    ap.add_argument("--hyp", default=os.path.join(ROOT, "train", "hyp_surveillance.yaml"),
                    help="augmentation/optimiser preset (hyp_surveillance / hyp_topdown / hyp_tuned)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32, help="T4: 32 @640, 16 @960 for v8n; -1 = auto")
    ap.add_argument("--device", default=None, help="0 / cpu / mps (default: auto)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--optimizer", default=None, help="SGD / AdamW / auto (default: from --hyp, else auto)")
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"],
                    help="ram is fastest; use disk if Colab RAM (~12 GB) runs out")
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a fraction of data (quick tests)")
    ap.add_argument("--seed", type=int, default=0)
    # callbacks
    ap.add_argument("--monitor", default="metrics/mAP50(B)",
                    help="metric for best checkpoint + early stopping, e.g. metrics/mAP50-95(B)")
    ap.add_argument("--patience", type=int, default=30, help="epochs without improvement before stopping")
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--target-map50", type=float, default=0.65)
    # output
    ap.add_argument("--project", default=os.path.join(ROOT, "runs", "train"),
                    help="put this on Google Drive in Colab so runs survive disconnects")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def load_hyp(path, optimizer_cli):
    with open(path) as f:
        hyp = yaml.safe_load(f) or {}
    optimizer = optimizer_cli or hyp.pop("optimizer", None) or "auto"
    hyp.pop("optimizer", None)
    if optimizer == "auto" and any(k in hyp for k in ("lr0", "momentum")):
        # optimizer=auto makes Ultralytics IGNORE lr0/momentum - tuned values would be silently lost
        print("NOTE: --hyp sets lr0/momentum, so using optimizer=SGD instead of 'auto'")
        optimizer = "SGD"
    return hyp, optimizer


def evaluate(weights, data, imgsz, split):
    m = YOLO(weights).val(data=data, imgsz=imgsz, split=split, batch=16, plots=split == "test",
                          verbose=False)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[m.names[int(c)]] = {"P": round(p, 4), "R": round(r, 4),
                                      "mAP50": round(ap50, 4), "mAP50-95": round(ap, 4)}
    return {"mAP50": round(m.box.map50, 4), "mAP50-95": round(m.box.map, 4),
            "P": round(m.box.mp, 4), "R": round(m.box.mr, 4), "per_class": per_class,
            "speed_ms": {k: round(v, 2) for k, v in m.speed.items()}}


def main():
    a = parse_args()
    name = a.name or "{}_{}".format(os.path.splitext(os.path.basename(a.model))[0], a.imgsz)
    run_dir = os.path.join(a.project, name)

    cb = MetricCheckpointEarlyStopping(monitor=a.monitor, patience=a.patience,
                                       min_delta=a.min_delta, target=a.target_map50
                                       if "mAP50(B)" in a.monitor else None)
    if a.resume:
        model = YOLO(os.path.join(run_dir, "weights", "last.pt"))
        cb.attach(model)
        model.train(resume=True)
    else:
        hyp, optimizer = load_hyp(a.hyp, a.optimizer)
        print("Preset {} | optimizer={} | {}".format(os.path.basename(a.hyp), optimizer, hyp))
        model = YOLO(a.model)
        cb.attach(model)
        model.train(
            data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            workers=a.workers, optimizer=optimizer, cache=False if a.cache == "none" else a.cache,
            fraction=a.fraction, seed=a.seed, deterministic=False,  # deterministic=True slows a T4 down
            amp=True, cos_lr=True,
            patience=0,          # built-in early stopping OFF - our callback handles it on mAP50
            save=True, save_period=-1, plots=True, val=True,
            project=a.project, name=name, exist_ok=True,
            **hyp
        )

    # ------------------------------------------------------------ final scoring
    best = os.path.join(run_dir, "weights", cb.filename)
    if not os.path.exists(best):
        best = os.path.join(run_dir, "weights", "best.pt")
    os.makedirs(os.path.join(ROOT, "weights"), exist_ok=True)
    dst = os.path.join(ROOT, "weights", name + ".pt")
    shutil.copy2(best, dst)

    results = {"weights": dst, "val": evaluate(dst, a.data, a.imgsz, "val")}
    with open(a.data) as f:
        has_test = "test" in (yaml.safe_load(f) or {})
    if has_test:
        results["test"] = evaluate(dst, a.data, a.imgsz, "test")
    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n{:<10}{:>8}{:>10}{:>7}{:>7}".format("split", "mAP50", "mAP50-95", "P", "R"))
    for split in ("val", "test"):
        if split in results:
            r = results[split]
            print("{:<10}{:>8.3f}{:>10.3f}{:>7.3f}{:>7.3f}".format(split, r["mAP50"], r["mAP50-95"], r["P"], r["R"]))
    key = "test" if has_test else "val"
    print("\nPer-class ({}):".format(key))
    for c, r in results[key]["per_class"].items():
        print("  {:<10} mAP50={:.3f}  mAP50-95={:.3f}  P={:.3f}  R={:.3f}".format(
            c, r["mAP50"], r["mAP50-95"], r["P"], r["R"]))
    score = results[key]["mAP50"]
    print("\nTarget mAP50 > {} on {}: {} ({:.3f})".format(
        a.target_map50, key, "PASS" if score > a.target_map50 else "NOT YET", score))
    print("Model -> {}\nCurves -> python train/plot_curves.py {}".format(dst, run_dir))


if __name__ == "__main__":
    main()
