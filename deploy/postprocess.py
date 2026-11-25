"""YOLOv8 raw-output decoding + NMS in pure numpy (no torch / no cv2 needed).

A YOLOv8 detect head exported to ONNX/TensorRT outputs shape (1, 4 + nc, A), where
A = 8400 anchors for 640x640 input. Rows 0-3 are (cx, cy, w, h) in *letterboxed input pixels*,
rows 4.. are per-class scores (already sigmoid-ed). There is no separate objectness score.
"""
import numpy as np

MAX_WH = 7680  # class offset used for batched (per-class) NMS


def xywh2xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_thres):
    """Greedy NMS. boxes: (N, 4) xyxy. Returns kept indices sorted by score desc."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def postprocess(output, ratio, pad, orig_shape, conf_thres=0.25, iou_thres=0.45,
                max_det=300, agnostic=False, max_candidates=3000):
    """Decode YOLOv8 output into (N, 6) [x1, y1, x2, y2, conf, cls] in original-image pixels.

    output:     raw model output, (1, 4+nc, A) or (4+nc, A) or (A, 4+nc)
    ratio, pad: from letterbox() - scale factor and (pad_left, pad_top)
    orig_shape: (h, w) of the original frame
    """
    pred = np.asarray(output, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:  # (4+nc, A) -> (A, 4+nc)
        pred = pred.T

    cls_scores = pred[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores[np.arange(len(cls_scores)), cls_ids]

    mask = confs > conf_thres
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)
    boxes, confs, cls_ids = pred[mask, :4], confs[mask], cls_ids[mask]

    if len(confs) > max_candidates:
        top = confs.argsort()[::-1][:max_candidates]
        boxes, confs, cls_ids = boxes[top], confs[top], cls_ids[top]

    xyxy = xywh2xyxy(boxes)
    offset = 0 if agnostic else cls_ids[:, None].astype(np.float32) * MAX_WH
    keep = nms(xyxy + offset, confs, iou_thres)[:max_det]
    xyxy, confs, cls_ids = xyxy[keep], confs[keep], cls_ids[keep]

    # Undo letterbox
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy /= ratio
    h, w = orig_shape[:2]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

    return np.concatenate([xyxy, confs[:, None], cls_ids[:, None].astype(np.float32)],
                          axis=1).astype(np.float32)
