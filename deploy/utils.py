"""Pre-processing, class-name loading and drawing helpers. Python 3.6+ compatible."""
import ast
import os

import cv2
import numpy as np

DEFAULT_NAMES = {0: "aircraft", 1: "drone", 2: "vehicle"}  # alphabetical, as Roboflow exports
PROJECT_YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "dataset.yaml")


# ----------------------------------------------------------------- preprocessing
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to new_shape. Returns (img, ratio, (pad_left, pad_top))."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / float(h), new_shape[1] / float(w))
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape[1] - new_w, new_shape[0] - new_h
    left, top = dw // 2, dh // 2
    img = cv2.copyMakeBorder(img, top, dh - top, left, dw - left,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(frame_bgr, imgsz=640, dtype=np.float32):
    """BGR HWC uint8 -> (1, 3, imgsz, imgsz) RGB float in [0, 1]."""
    img, ratio, pad = letterbox(frame_bgr, (imgsz, imgsz))
    blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    blob /= 255.0
    return blob[None].astype(dtype, copy=False), ratio, pad


# ----------------------------------------------------------------- class names
def load_names(source=None):
    """Load {id: name} from a dataset YAML, a 'a,b,c' string, a list, or a dict."""
    if source is None:  # raw TensorRT engines carry no names -> use the project's dataset.yaml
        return load_names(PROJECT_YAML) if os.path.isfile(PROJECT_YAML) else dict(DEFAULT_NAMES)
    if isinstance(source, dict):
        return {int(k): str(v) for k, v in source.items()}
    if isinstance(source, (list, tuple)):
        return {i: str(n) for i, n in enumerate(source)}
    if isinstance(source, str) and os.path.isfile(source):
        import yaml
        with open(source) as f:
            data = yaml.safe_load(f)
        return load_names(data.get("names", data))
    if isinstance(source, str):
        try:  # ultralytics metadata stores names as a dict repr: "{0: 'drone', ...}"
            return load_names(ast.literal_eval(source))
        except (ValueError, SyntaxError):
            return {i: n.strip() for i, n in enumerate(source.split(","))}
    raise ValueError("Cannot parse class names from {!r}".format(source))


# ----------------------------------------------------------------- drawing
def color_for(idx):
    """Deterministic, well-spread BGR colour for an integer id."""
    idx = int(idx) * 37 + 11
    return (int((idx * 67) % 200 + 55), int((idx * 113) % 200 + 55), int((idx * 29) % 200 + 55))


def _label(img, text, x1, y1, color):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_top = y1 - th - base - 2
    if y_top < 0:          # no room above the box (object at the top edge): put the label inside it
        y_top = y1 + 2
    x1 = min(max(x1, 0), max(img.shape[1] - tw - 4, 0))
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 4, y_top + th + base + 2), color, -1)
    cv2.putText(img, text, (x1 + 2, y_top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)


def draw_detections(img, dets, names):
    for x1, y1, x2, y2, conf, cls in dets:
        c = color_for(cls)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, c, 2)
        _label(img, "{} {:.2f}".format(names.get(int(cls), int(cls)), conf), p1[0], p1[1], c)
    return img


def draw_tracks(img, tracks, names, trails=None):
    """tracks: (N, 7) [x1, y1, x2, y2, id, score, cls]; trails: {id: [(cx, cy), ...]}."""
    for x1, y1, x2, y2, tid, score, cls in tracks:
        c = color_for(tid)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, c, 2)
        _label(img, "#{} {} {:.2f}".format(int(tid), names.get(int(cls), int(cls)), score),
               p1[0], p1[1], c)
        if trails and int(tid) in trails and len(trails[int(tid)]) > 1:
            pts = np.asarray(trails[int(tid)], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, c, 2, cv2.LINE_AA)
    return img


def draw_lost(img, lost, color=(160, 160, 160)):
    """Occluded tracks: dashed grey box at the Kalman-PREDICTED position, labelled with the id."""
    for x1, y1, x2, y2, tid, _, _ in lost:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
            n = max(int(np.hypot(bx - ax, by - ay) // 8), 1)
            for k in range(0, n, 2):  # 8 px dash, 8 px gap
                p = (int(ax + (bx - ax) * k / float(n)), int(ay + (by - ay) * k / float(n)))
                q = (int(ax + (bx - ax) * (k + 1) / float(n)), int(ay + (by - ay) * (k + 1) / float(n)))
                cv2.line(img, p, q, color, 1, cv2.LINE_AA)
        cv2.putText(img, "#{} lost".format(int(tid)), (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    return img


def draw_hud(img, lines):
    width = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for l in lines)
    x2, y2 = min(width + 20, img.shape[1]), min(24 * len(lines) + 8, img.shape[0])
    roi = img[0:y2, 0:x2]
    roi[:] = (roi * 0.45).astype(img.dtype)  # translucent panel: objects underneath stay visible
    y = 22
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 1, cv2.LINE_AA)
        y += 24
    return img


# ----------------------------------------------------------------- video sources
from .camera import gstreamer_csi, open_source  # noqa: E402,F401  (kept here for backwards compatibility)
