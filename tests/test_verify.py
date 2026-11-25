import numpy as np

from export.verify import judge, match_detections, precision_of


def test_match_detections_same_class_one_to_one():
    ref = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 40, 40, 0.8, 1]], dtype=np.float32)
    cand = np.array([[1, 0, 11, 10, 0.88, 0],      # IoU 0.82 with ref[0] -> left over as extra
                     [20, 20, 40, 40, 0.8, 2],     # right place, WRONG class -> no match
                     [0, 0, 10, 10, 0.5, 0]], dtype=np.float32)  # IoU 1.0 -> wins ref[0] (greedy)
    n_ref, n_cand, ious, sdiff = match_detections(ref, cand)
    assert (n_ref, n_cand, len(ious)) == (2, 3, 1)
    assert abs(ious[0] - 1.0) < 1e-6 and abs(sdiff[0] - 0.4) < 1e-6


def test_judge_and_precision_parsing():
    assert precision_of("w/yolo_custom_int8.engine") == "int8"
    assert precision_of("w/yolo_custom.pt@fp16") == "fp16"
    assert precision_of("w/yolo_custom.onnx") == "fp32"
    good = {"match_rate": 0.97, "mean_iou": 0.98, "mean_score_diff": 0.01}
    assert judge(good, "fp16") == "PASS" and judge(good, "fp32") == "CHECK"
    assert judge({"match_rate": None}, "fp32") == "NO-DETS"
