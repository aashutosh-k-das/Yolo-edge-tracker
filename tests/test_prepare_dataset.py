import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))
from prepare_dataset import original_stem, parse_label  # noqa: E402


def test_original_stem_strips_roboflow_suffix():
    assert original_stem("clip07_000123_jpg.rf.3f9a1b2c4d") == "clip07_000123"
    assert original_stem("IMG_0001_png.rf.ABC123") == "IMG_0001"
    assert original_stem("plain_name") == "plain_name"


def test_polygon_converted_to_box(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("1 0.2 0.2 0.4 0.2 0.4 0.6 0.2 0.6\n0 0.5 0.5 0.1 0.1\n7 0.5 0.5 0.1 0.1\n")
    rows, errs, n_poly = parse_label(str(p), nc=3)
    assert n_poly == 1 and len(errs) == 1          # class 7 out of range
    c, cx, cy, w, h = rows[0]
    assert c == 1 and abs(cx - 0.3) < 1e-9 and abs(cy - 0.4) < 1e-9
    assert abs(w - 0.2) < 1e-9 and abs(h - 0.4) < 1e-9


def _fake_roboflow_export(root):
    """3 videos x 4 frames, each frame also has an augmented copy, spread over RF's splits."""
    root.mkdir(parents=True)
    (root / "data.yaml").write_text("nc: 3\nnames: ['aircraft', 'drone', 'vehicle']\n")
    (root / "README.roboflow.txt").write_text("exported via roboflow")
    splits = ["train", "valid", "test"]
    i = 0
    for vid in range(3):
        for frame in range(4):
            for aug in ("aaa", "bbb"):
                split = splits[i % 3]
                i += 1
                stem = "clip{}_{:04d}_jpg.rf.{}{}{}".format(vid, frame, aug, vid, frame)
                (root / split / "images").mkdir(parents=True, exist_ok=True)
                (root / split / "labels").mkdir(parents=True, exist_ok=True)
                (root / split / "images" / (stem + ".jpg")).write_bytes(b"\xff\xd8fake")
                (root / split / "labels" / (stem + ".txt")).write_text("{} 0.5 0.5 0.1 0.1\n".format(vid))


def test_roboflow_resplit_keeps_videos_together(tmp_path):
    rf = tmp_path / "rf"
    _fake_roboflow_export(rf)
    out = tmp_path / "ds" / "dataset"
    subprocess.check_call([sys.executable, os.path.join(ROOT, "data", "prepare_dataset.py"),
                           "--roboflow", str(rf), "--out", str(out),
                           "--group-pattern", r"^(.*)_\d+$", "--split", "0.34", "0.33", "0.33"])
    cfg = yaml.safe_load((tmp_path / "ds" / "dataset.yaml").read_text())
    assert cfg["names"] == {0: "aircraft", 1: "drone", 2: "vehicle"}

    video_split = {}
    for part in ("train", "val", "test"):
        for fn in os.listdir(str(out / "images" / part)):
            vid = fn.split("_")[0]
            assert video_split.setdefault(vid, part) == part, "video {} leaked across splits".format(vid)
            assert (out / "labels" / part / (os.path.splitext(fn)[0] + ".txt")).exists()
    assert len(video_split) == 3
