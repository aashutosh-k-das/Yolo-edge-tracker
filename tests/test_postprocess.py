import numpy as np

from deploy.postprocess import nms, postprocess


def _raw(nc=3, anchors=8400):
    return np.zeros((1, 4 + nc, anchors), dtype=np.float32)


def _put(raw, idx, cx, cy, w, h, cls, score):
    raw[0, :4, idx] = [cx, cy, w, h]
    raw[0, 4 + cls, idx] = score


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_postprocess_decodes_and_unletterboxes():
    # Original 1280x720 frame -> letterboxed to 640x640: ratio 0.5, pad_top 140.
    ratio, pad = 0.5, (0, 140)
    raw = _raw()
    _put(raw, 10, 320, 320, 100, 50, cls=0, score=0.9)    # drone, centre of image
    _put(raw, 11, 322, 321, 100, 50, cls=0, score=0.6)    # duplicate -> removed by NMS
    _put(raw, 12, 322, 321, 100, 50, cls=1, score=0.7)    # same place, other class -> kept
    _put(raw, 13, 100, 200, 20, 20, cls=2, score=0.1)     # below conf -> dropped
    dets = postprocess(raw, ratio, pad, (720, 1280), conf_thres=0.25, iou_thres=0.45)

    assert dets.shape == (2, 6)
    d0 = dets[dets[:, 5] == 0][0]
    np.testing.assert_allclose(d0[:4], [540, 310, 740, 410], atol=1e-3)
    assert abs(d0[4] - 0.9) < 1e-6
    assert set(dets[:, 5].astype(int)) == {0, 1}


def test_postprocess_agnostic_and_empty():
    raw = _raw()
    _put(raw, 0, 320, 320, 100, 50, cls=0, score=0.9)
    _put(raw, 1, 322, 321, 100, 50, cls=1, score=0.7)
    assert len(postprocess(raw, 1.0, (0, 0), (640, 640), agnostic=True)) == 1
    assert postprocess(_raw(), 1.0, (0, 0), (640, 640)).shape == (0, 6)


def test_postprocess_clips_to_image():
    raw = _raw()
    _put(raw, 0, 5, 5, 40, 40, cls=0, score=0.9)
    d = postprocess(raw, 1.0, (0, 0), (640, 640))
    assert d[0, 0] == 0 and d[0, 1] == 0
