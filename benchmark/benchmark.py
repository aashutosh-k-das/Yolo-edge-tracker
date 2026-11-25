"""Phase 6a - latency / FPS benchmark: PyTorch vs ONNX Runtime vs TensorRT (runs on Jetson too).

    python -m benchmark.benchmark --images data/dataset/images/test --plot --weights \
        weights/yolo_custom.pt weights/yolo_custom.pt@fp16 weights/yolo_custom.onnx \
        weights/yolo_custom_fp16.engine weights/yolo_custom_int8.engine

The first model is the baseline for the speed-up column. Every backend uses the SAME
letterbox + decode + NMS code, so the differences come from the inference engine alone.
'best.pt@fp16' = PyTorch half precision on CUDA. Models that cannot load on this device
(e.g. .pt on a Python 3.6 Jetson) are skipped with a message.

Measures preprocess / inference / postprocess / tracking time separately on REAL images
(random noise gives almost no detections and under-reports NMS + tracking cost).
Tip (Jetson): run `sudo nvpmodel -m 0 && sudo jetson_clocks` first, and log power with
`tegrastats --interval 500 --logfile tegrastats.log` in another terminal.
"""
import argparse
import csv
import glob
import json
import os
import platform
import time

import cv2
import numpy as np

from deploy.detectors import load_detector
from track import BYTETracker


def device_name():
    try:
        with open("/proc/device-tree/model") as f:  # Jetson
            return f.read().strip("\x00").strip()
    except IOError:
        return "{} {}".format(platform.system(), platform.machine())


def load_frames(images=None, video=None, n=100):
    frames = []
    if images:
        for p in sorted(glob.glob(os.path.join(images, "*")))[:n]:
            img = cv2.imread(p)
            if img is not None:
                frames.append(img)
    elif video:
        # sample evenly across the whole clip, not just its first n frames (often empty scenery)
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(total // n, 1) if total > 0 else 1
        i = 0
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(f)
            i += 1
        cap.release()
    if not frames:
        print("WARNING: no images/video given - using random 1280x720 frames (few detections).")
        rng = np.random.RandomState(0)
        frames = [rng.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(10)]
    return frames


def precision_of(path):
    name = os.path.basename(path).lower()
    if name.endswith("@fp16"):
        return "fp16"
    for p in ("int8", "fp16", "fp32"):
        if p in name:
            return p
    return "fp32"


def bench(weights, frames, imgsz, warmup, iters, track, det=None, progress=None):
    """Time one model. `det`: an already-loaded detector (e.g. cached by the dashboard), else loaded here.
    `progress`: optional callback(fraction)."""
    own = det is None
    if own:
        det = load_detector(weights, imgsz=imgsz)
    det.conf = 0.1 if track else 0.25
    tracker = BYTETracker() if track else None
    for i in range(warmup):
        det(frames[i % len(frames)])

    rec = {k: [] for k in ("pre", "infer", "post", "track", "total")}
    ndet = []
    t_wall = time.perf_counter()
    for i in range(iters):
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        dets = det(f)
        t1 = time.perf_counter()
        if tracker:
            tracker.update(dets)
        t2 = time.perf_counter()
        for k in ("pre", "infer", "post"):
            rec[k].append(det.timings[k])
        rec["track"].append((t2 - t1) * 1e3)
        rec["total"].append((t2 - t0) * 1e3)
        ndet.append(len(dets))
        if progress and i % 10 == 0:
            progress((i + 1) / float(iters))
    wall = time.perf_counter() - t_wall
    if own:
        det.close()

    tot = np.array(rec["total"])
    return {
        "model": os.path.basename(weights),
        "backend": det.backend,
        "precision": precision_of(weights),
        "imgsz": det.imgsz,
        "size_mb": round(os.path.getsize(weights.replace("@fp16", "")) / 1e6, 2),
        "pre_ms": round(float(np.mean(rec["pre"])), 2),
        "infer_ms": round(float(np.mean(rec["infer"])), 2),
        "post_ms": round(float(np.mean(rec["post"])), 2),
        "track_ms": round(float(np.mean(rec["track"])), 2),
        "total_ms": round(float(tot.mean()), 2),
        "p50_ms": round(float(np.percentile(tot, 50)), 2),
        "p95_ms": round(float(np.percentile(tot, 95)), 2),
        "fps": round(iters / wall, 1),
        "avg_dets": round(float(np.mean(ndet)), 1),
        "device": device_name(),
        "tracking": bool(track),
    }


def print_table(rows):
    cols = ["model", "backend", "precision", "size_mb", "pre_ms", "infer_ms", "post_ms",
            "track_ms", "total_ms", "p95_ms", "fps", "speedup"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--track", action="store_true", help="include ByteTrack in the timed loop")
    ap.add_argument("--out", default="benchmark/results/speed.csv")
    ap.add_argument("--plot", action="store_true", help="save a latency-breakdown chart next to --out")
    a = ap.parse_args()

    frames = load_frames(a.images, a.video)
    print("Device: {} | {} frames | {} iters".format(device_name(), len(frames), a.iters))
    rows = []
    for w in a.weights:
        print("Benchmarking", w, "...")
        try:
            rows.append(bench(w, frames, a.imgsz, a.warmup, a.iters, a.track))
        except Exception as e:  # e.g. no torch on the Jetson, engine built for another GPU
            print("  SKIPPED {}: {}: {}".format(w, type(e).__name__, e))
    if not rows:
        raise SystemExit("No model could be benchmarked.")
    base = rows[0]["total_ms"]
    for r in rows:
        r["speedup"] = "{:.2f}x".format(base / r["total_ms"])
    print_table(rows)
    print("speedup = end-to-end latency of {} / latency of each model".format(rows[0]["model"]))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            wr.writeheader()
        wr.writerows(rows)
    with open(os.path.splitext(a.out)[0] + "_latest.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nAppended results to", a.out)
    if a.plot:
        plot_breakdown(rows, os.path.splitext(a.out)[0] + "_breakdown.png")


# Stage colours = categorical slots 1-4 (identity of the pipeline stage).
STAGES = [("pre_ms", "preprocess", "#2a78d6"), ("infer_ms", "inference", "#eb6834"),
          ("post_ms", "decode + NMS", "#1baf7a"), ("track_ms", "ByteTrack", "#eda100")]


def plot_breakdown(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["{}\n{}".format(r["model"], r["backend"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(rows) + 1.6), dpi=140)
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for key, name, color in STAGES:
        vals = np.array([r[key] for r in rows])
        if not vals.any():
            continue
        ax.barh(y, vals, left=left, height=0.55, color=color, edgecolor="white", linewidth=2, label=name)
        left += vals
    for yi, r, l in zip(y, rows, left):
        ax.text(l, yi, "  {:.1f} ms | {:.0f} FPS".format(r["total_ms"], r["fps"]), va="center",
                fontsize=8, color="#1f1f1e")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color="#1f1f1e")
    ax.set_xlabel("mean latency per frame (ms)", fontsize=9, color="#6b6a64")
    ax.set_xlim(0, left.max() * 1.3)
    ax.set_title("Latency breakdown on {}".format(rows[0]["device"]), loc="left", fontsize=11, color="#1f1f1e", pad=26)
    ax.grid(True, axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#e6e5e0")
    ax.tick_params(colors="#6b6a64", labelsize=8, length=0)
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0),
              borderaxespad=0.2, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(path)
    print("Saved", path)


if __name__ == "__main__":
    main()
