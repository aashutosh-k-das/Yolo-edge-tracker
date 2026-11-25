"""Create the README images in docs/assets/ from your own model, video and results.

    python docs/make_assets.py --weights weights/yolo_custom.onnx \
        --images data/dataset/images/test --video out/demo_tracked.mp4 \
        --train-run runs/train/v8n_640

    demo.gif          first seconds of an annotated tracking video (from deploy.run --save)
    samples.jpg       2x3 grid of test images with detections
    curves.png        training curves (train/plot_curves.py output)
    latency.png       latency breakdown (benchmark.py --plot output)
    tradeoff.png      speed vs accuracy (benchmark/report.py output)
    confusion.png     normalised confusion matrix of the training run
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")


def make_gif(video, out, seconds=6.0, width=640, fps=10):
    from PIL import Image
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(src_fps / fps)), 1)
    frames, i = [], 0
    while len(frames) < seconds * fps:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            f = cv2.resize(f, (width, int(h * width / float(w))), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(f[:, :, ::-1]).quantize(colors=128))
        i += 1
    cap.release()
    if not frames:
        raise SystemExit("could not read " + video)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print("wrote {} ({} frames, {:.1f} MB)".format(out, len(frames), os.path.getsize(out) / 1e6))


def make_grid(weights, images, out, n=6, cols=3, tile=(480, 320)):
    import sys
    sys.path.insert(0, ROOT)
    from deploy.detectors import load_detector
    from deploy.utils import draw_detections
    det = load_detector(weights)
    files = sorted(glob.glob(os.path.join(images, "*")))
    # prefer images with several detections - more informative than empty sky
    scored = []
    for p in files[:200]:
        img = cv2.imread(p)
        if img is None:
            continue
        d = det(img)
        scored.append((len(set(d[:, 5].astype(int).tolist())), len(d), p, img, d))
    scored.sort(key=lambda s: (-s[0], -s[1]))
    tiles = []
    for _, _, _, img, d in scored[:n]:
        vis = draw_detections(img.copy(), d, det.names)
        h, w = vis.shape[:2]
        r = min(tile[0] / float(w), tile[1] / float(h))
        vis = cv2.resize(vis, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
        canvas = np.full((tile[1], tile[0], 3), 255, np.uint8)
        y, x = (tile[1] - vis.shape[0]) // 2, (tile[0] - vis.shape[1]) // 2
        canvas[y:y + vis.shape[0], x:x + vis.shape[1]] = vis
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.full((tile[1], tile[0], 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    det.close()
    print("wrote", out)


def copy_if(src, name):
    if src and os.path.exists(src):
        shutil.copy2(src, os.path.join(ASSETS, name))
        print("copied {} -> docs/assets/{}".format(os.path.relpath(src, ROOT), name))
    else:
        print("missing {} (skipped)".format(src))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--images", default=os.path.join(ROOT, "data", "dataset", "images", "test"))
    ap.add_argument("--video", help="annotated video from `python -m deploy.run ... --save`")
    ap.add_argument("--train-run", help="e.g. runs/train/v8n_640")
    ap.add_argument("--results", default=os.path.join(ROOT, "benchmark", "results"))
    a = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)

    if a.video:
        make_gif(a.video, os.path.join(ASSETS, "demo.gif"))
    if a.weights and os.path.isdir(a.images):
        make_grid(a.weights, a.images, os.path.join(ASSETS, "samples.jpg"))
    if a.train_run:
        copy_if(os.path.join(a.train_run, "curves.png"), "curves.png")
        copy_if(os.path.join(a.train_run, "confusion_matrix_normalized.png"), "confusion.png")
    copy_if(os.path.join(a.results, "speed_breakdown.png"), "latency.png")
    copy_if(os.path.join(a.results, "tradeoff.png"), "tradeoff.png")


if __name__ == "__main__":
    main()
