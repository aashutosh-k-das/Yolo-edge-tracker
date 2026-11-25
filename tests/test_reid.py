import numpy as np
import pytest

from track import BYTETracker, tracks_to_array


def _det(cx, cy, w, h, score=0.9, cls=1):
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, score, cls]


def _run(tracker, frames_dets, frames=None):
    ids = set()
    for i, dets in enumerate(frames_dets):
        f = frames[i] if frames is not None else None
        out = tracks_to_array(tracker.update(np.array(dets, dtype=np.float64).reshape(-1, 6), f))
        ids.update(out[:, 4].astype(int).tolist())
    return ids


def _hover_scenario():
    """Drone flies right at 6 px/frame, hides 20 frames, re-appears hovering where it vanished.
    The Kalman prediction has drifted ~120 px ahead -> IoU with the real box is 0."""
    seq = [[_det(100 + 6 * f, 200, 40, 30)] for f in range(20)]
    seq += [[] for _ in range(20)]
    seq += [[_det(100 + 6 * 19, 200, 40, 30)] for _ in range(10)]
    return seq


def test_hovering_drone_gets_new_id_without_reid():
    assert len(_run(BYTETracker(reid=False), _hover_scenario())) == 2


def test_hovering_drone_keeps_id_with_reid():
    trk = BYTETracker(reid=True)
    assert _run(trk, _hover_scenario()) == {1}
    assert trk.reid_events >= 1


def test_fast_small_drone():
    """12x12 px drone moving 20 px/frame: consecutive boxes never overlap (IoU = 0).
    Plain ByteTrack loses it after frame 1 (every new box is an unconfirmed track that is never
    confirmed); re-ID keeps following it with one id."""
    seq = [[_det(50 + 20 * f, 300, 12, 12)] for f in range(30)]

    def coverage(trk):
        n, ids = 0, set()
        for dets in seq:
            out = tracks_to_array(trk.update(np.array(dets)))
            n += len(out) > 0
            ids.update(out[:, 4].astype(int).tolist())
        return n, ids

    n_plain, _ = coverage(BYTETracker(reid=False))
    n_reid, ids = coverage(BYTETracker(reid=True))
    assert n_plain <= 2
    assert n_reid == 30 and ids == {1}


def _reappear(cls_after=1):
    """Stationary object, hidden 5 frames, re-appears 45 px to the right (no IoU overlap)."""
    seq = [[_det(200, 200, 40, 30, cls=1)] for _ in range(5)] + [[] for _ in range(5)]
    return seq + [[_det(245, 200, 40, 30, cls=cls_after)] for _ in range(5)]


def test_reid_respects_class():
    assert _run(BYTETracker(reid=True), _reappear(cls_after=1)) == {1}
    assert _run(BYTETracker(reid=True), _reappear(cls_after=0)) == {1, 2}


def _frame_with_box(color, box):
    f = np.full((400, 400, 3), 200, np.uint8)
    x1, y1, x2, y2 = box
    f[y1:y2, x1:x2] = color
    return f


def test_appearance_gate():
    pytest.importorskip("cv2")
    red, blue = (0, 0, 220), (220, 60, 0)
    blank = np.full((400, 400, 3), 200, np.uint8)
    before, after = (180, 185, 220, 215), (225, 185, 265, 215)

    def frames(color_after):
        return [_frame_with_box(red, before)] * 5 + [blank] * 5 + [_frame_with_box(color_after, after)] * 5

    assert _run(BYTETracker(reid=True), _reappear(), frames(red)) == {1}       # same object -> same id
    assert _run(BYTETracker(reid=True), _reappear(), frames(blue)) == {1, 2}   # different look -> new id
    assert _run(BYTETracker(reid=True), _reappear(), None) == {1}              # no frame: motion only
