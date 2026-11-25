"""Lightweight appearance features for re-identification (no extra neural network).

An HSV colour histogram of the central part of each box: ~0.05 ms per box on a Jetson Nano,
versus several ms for a deep ReID CNN. It is good at telling a white airliner from a dark
quadcopter or a red car, but weak on tiny boxes, so boxes under `min_size` px return None and
the tracker falls back to motion-only matching for them.
"""
import numpy as np


def hsv_hist(frame, tlbr, min_size=10, inner=0.8, bins=(16, 8)):
    """L2-normalised Hue-Saturation histogram of the central `inner` fraction of a box, or None."""
    import cv2
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = tlbr
    bw, bh = x2 - x1, y2 - y1
    if bw < min_size or bh < min_size:
        return None
    m = (1.0 - inner) / 2  # shrink towards the centre: less sky / road background
    x1, x2 = int(max(x1 + m * bw, 0)), int(min(x2 - m * bw, w))
    y1, y2 = int(max(y1 + m * bh, 0)), int(min(y2 - m * bh, h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256]).flatten()
    n = np.linalg.norm(hist)
    return hist / n if n > 0 else None


def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b))
