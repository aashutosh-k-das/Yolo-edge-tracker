"""IoU cost matrices and Hungarian assignment for ByteTrack (numpy + scipy only)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a, b):
    """IoU between two sets of xyxy boxes. a: (N, 4), b: (M, 4) -> (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def iou_distance(atracks, btracks):
    """1 - IoU between two lists of STrack objects."""
    a = [t.tlbr for t in atracks]
    b = [t.tlbr for t in btracks]
    return 1.0 - box_iou(a, b)


def fuse_score(cost, detections):
    """Weight IoU similarity by detection confidence (ByteTrack's default for non-MOT20)."""
    if cost.size == 0:
        return cost
    scores = np.array([d.score for d in detections])[None, :]
    return 1.0 - (1.0 - cost) * scores


def linear_assignment(cost, thresh):
    """Hungarian matching; pairs with cost > thresh are rejected.

    Returns (matches (K, 2), unmatched_rows, unmatched_cols).
    """
    n, m = cost.shape
    if cost.size == 0:
        return np.empty((0, 2), dtype=int), list(range(n)), list(range(m))
    # Make invalid pairs prohibitively expensive so they never displace valid ones.
    c = np.where(cost > thresh, 1e5, cost)
    rows, cols = linear_sum_assignment(c)
    keep = c[rows, cols] <= thresh
    matches = np.stack([rows[keep], cols[keep]], axis=1)
    unmatched_rows = sorted(set(range(n)) - set(matches[:, 0].tolist()))
    unmatched_cols = sorted(set(range(m)) - set(matches[:, 1].tolist()))
    return matches, unmatched_rows, unmatched_cols
