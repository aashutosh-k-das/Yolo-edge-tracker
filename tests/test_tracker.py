import numpy as np

from track import BYTETracker, tracks_to_array
from track.matching import box_iou, linear_assignment


def _box(cx, cy, w=40, h=30):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_box_iou():
    iou = box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou[0], [1.0, 1 / 3.0, 0.0], atol=1e-6)


def test_linear_assignment_rejects_above_threshold():
    cost = np.array([[0.1, 0.9], [0.95, 0.99]])
    matches, ur, uc = linear_assignment(cost, 0.8)
    assert matches.tolist() == [[0, 0]]
    assert ur == [1] and uc == [1]


def test_two_objects_keep_ids_through_occlusion_and_low_scores():
    rng = np.random.RandomState(0)
    trk = BYTETracker(track_thresh=0.5, low_thresh=0.1, match_thresh=0.8, track_buffer=30)
    seen = set()
    for f in range(60):
        dets = []
        # Object A moves right; scores dip to 0.3 for frames 20-29 (blur) -> rescued by stage 2.
        sa = 0.3 if 20 <= f < 30 else 0.9
        dets.append(_box(100 + 4 * f, 200) + [sa, 0])
        # Object B moves down; fully occluded (no detection) for frames 35-44.
        if not 35 <= f < 45:
            dets.append(_box(400, 100 + 3 * f) + [0.85, 1])
        dets = np.array(dets) + np.r_[rng.normal(0, 1, 4), 0, 0]
        out = tracks_to_array(trk.update(dets))
        seen.update(out[:, 4].astype(int).tolist())
        if f >= 1 and not 35 <= f < 45:
            assert len(out) == 2, "frame {}: {} tracks".format(f, len(out))
    assert seen == {1, 2}, "IDs switched: {}".format(seen)


def test_new_track_needs_confirmation():
    trk = BYTETracker()
    trk.update(np.array([_box(50, 50) + [0.9, 0]]))      # frame 1: activated immediately
    out = trk.update(np.array([_box(52, 50) + [0.9, 0],
                               _box(300, 300) + [0.9, 2]]))  # new object appears
    assert len(out) == 1                                   # not confirmed yet
    out = trk.update(np.array([_box(54, 50) + [0.9, 0],
                               _box(302, 300) + [0.9, 2]]))
    assert sorted(t.track_id for t in out) == [1, 2]


def test_empty_frames():
    trk = BYTETracker()
    for _ in range(5):
        assert trk.update(np.zeros((0, 6))) == []


def test_class_majority_vote_stabilises_flickering_labels():
    trk = BYTETracker()
    classes = [2, 2, 7, 2, 2, 5, 2, 2, 7, 2]          # car, car, truck, car, ... (detector flicker)
    for f, c in enumerate(classes):
        out = tracks_to_array(trk.update(np.array([_box(100 + 2 * f, 100) + [0.9, c]])))
    assert len(out) == 1 and int(out[0, 6]) == 2      # reported as 'car' despite the last flickers
