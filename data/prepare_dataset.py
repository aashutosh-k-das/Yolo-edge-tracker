"""Phase 1 - validate a YOLO-format dataset, print statistics, split it, and write dataset.yaml.

Roboflow export ("YOLOv8" format, unzipped to data/raw/roboflow):
    python data/prepare_dataset.py --roboflow data/raw/roboflow --dry-run          # stats only
    python data/prepare_dataset.py --roboflow data/raw/roboflow --group-pattern "^(.*)_\\d+$"

  * class names + order are read from Roboflow's data.yaml (Roboflow sorts them alphabetically)
  * Roboflow's "_jpg.rf.<hash>" filename suffix is stripped before grouping, so augmented
    copies of one source image always stay in the same split
  * polygon labels (Smart Polygon tool) are converted to bounding boxes

Any other YOLO export (CVAT / LabelImg):
    python data/prepare_dataset.py --images raw/images --labels raw/labels --names aircraft drone vehicle

--group-pattern: regex whose group(1) is a "source" key (e.g. video name). Every group goes to
exactly ONE split, which prevents near-identical video frames leaking from train into val/test.
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE_BINS = [("tiny <16px", 16), ("small <32px", 32), ("medium <96px", 96), ("large", 1e9)]
DEFAULT_NAMES = ["aircraft", "drone", "vehicle"]  # alphabetical = Roboflow's order
RF_SUFFIX = re.compile(r"_(jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-zA-Z]+$", re.IGNORECASE)


def original_stem(stem):
    """'clip07_000123_jpg.rf.3f9a...' -> 'clip07_000123' (Roboflow renames every file)."""
    return RF_SUFFIX.sub("", stem)


def find_files(root, exts):
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in exts or fn.lower().startswith("readme"):
                continue
            if stem in out:
                print("  WARNING duplicate stem '{}' - keeping first".format(stem))
                continue
            out[stem] = os.path.join(dp, fn)
    return out


def parse_label(path, nc):
    """Return (rows, errors, n_polygons_converted). rows = [(cls, cx, cy, w, h)]."""
    rows, errors, n_poly = [], [], 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            try:
                c = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                errors.append("line {}: non-numeric".format(ln))
                continue
            if len(vals) == 4:
                cx, cy, w, h = vals
            elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon x1 y1 x2 y2 ... -> bbox
                xs, ys = vals[0::2], vals[1::2]
                x1, x2 = max(min(xs), 0.0), min(max(xs), 1.0)
                y1, y2 = max(min(ys), 0.0), min(max(ys), 1.0)
                cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                n_poly += 1
            else:
                errors.append("line {}: expected 5 values, got {}".format(ln, len(parts)))
                continue
            if not 0 <= c < nc:
                errors.append("line {}: class {} out of range [0, {})".format(ln, c, nc))
                continue
            if w <= 0 or h <= 0 or not all(0 <= v <= 1.0 + 1e-6 for v in (cx, cy, w, h)):
                errors.append("line {}: coords not normalised / zero-size {}".format(ln, parts[1:5]))
                continue
            rows.append((c, cx, cy, w, h))
    return rows, errors, n_poly


def split_groups(keys, ratios, seed):
    keys = sorted(keys)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return {"train": keys[:n_train], "val": keys[n_train:n_train + n_val],
            "test": keys[n_train + n_val:]}


def resolve_names(a):
    if a.names:
        return a.names
    if a.roboflow:
        rf_yaml = os.path.join(a.roboflow, "data.yaml")
        if os.path.isfile(rf_yaml):
            with open(rf_yaml) as f:
                names = yaml.safe_load(f)["names"]
            names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
            print("Class names from Roboflow data.yaml:", names)
            return names
    return DEFAULT_NAMES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roboflow", default=None, help="unzipped Roboflow YOLOv8 export folder")
    ap.add_argument("--images", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--names", nargs="+", default=None,
                    help="class names IN ID ORDER (default: Roboflow data.yaml, else %s)" % DEFAULT_NAMES)
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-pattern", default=None,
                    help="regex; group(1) of the (original) image stem = group kept within one split")
    ap.add_argument("--imgsz", type=int, default=640, help="training size, for box-size stats")
    ap.add_argument("--symlink", action="store_true", help="symlink instead of copying")
    ap.add_argument("--dry-run", action="store_true", help="only validate + print stats")
    a = ap.parse_args()

    a.images = a.images or a.roboflow
    a.labels = a.labels or a.roboflow
    if not (a.images and a.labels):
        ap.error("give --roboflow DIR, or both --images and --labels")
    names = resolve_names(a)
    nc = len(names)

    images = find_files(a.images, IMG_EXTS)
    labels = find_files(a.labels, {".txt"})
    print("Found {} images, {} label files".format(len(images), len(labels)))

    orphans = sorted(set(labels) - set(images))
    if orphans:
        print("  {} label files have no image (ignored), e.g. {}".format(len(orphans), orphans[:3]))

    inst, img_per_cls, sizes = Counter(), Counter(), Counter()
    clean_labels, bad_files, background, polys = {}, {}, 0, 0
    for stem in images:
        if stem not in labels:
            background += 1
            clean_labels[stem] = []
            continue
        rows, errs, n_poly = parse_label(labels[stem], nc)
        polys += n_poly
        if errs:
            bad_files[stem] = errs
        clean_labels[stem] = rows
        if not rows:
            background += 1
        for c, _, _, w, h in rows:
            inst[c] += 1
            px = (w * h) ** 0.5 * a.imgsz
            sizes[next(name for name, lim in SIZE_BINS if px < lim)] += 1
        for c in set(r[0] for r in rows):
            img_per_cls[c] += 1

    # ---------------------------------------------------------------- report
    print("\n=== Class balance ===")
    print("{:<4}{:<12}{:>10}{:>10}".format("id", "class", "boxes", "images"))
    for i, n in enumerate(names):
        print("{:<4}{:<12}{:>10}{:>10}".format(i, n, inst[i], img_per_cls[i]))
    print("background images (no boxes): {} ({:.1f}%)".format(background, 100.0 * background / max(len(images), 1)))
    if polys:
        print("converted {} polygon labels to bounding boxes".format(polys))

    total = sum(sizes.values()) or 1
    print("\n=== Box size at imgsz={} (approx., sqrt(area)) ===".format(a.imgsz))
    for name, _ in SIZE_BINS:
        print("  {:<14}{:>7}  ({:.1f}%)".format(name, sizes[name], 100.0 * sizes[name] / total))
    if (sizes["tiny <16px"] + sizes["small <32px"]) / float(total) > 0.3:
        print("  >> Many small objects (typical for drones). Consider --imgsz 960/1280 for training,"
              " or tiling/SAHI at inference.")

    if bad_files:
        print("\n=== {} label files with problems (invalid lines are DROPPED) ===".format(len(bad_files)))
        for stem, errs in list(bad_files.items())[:10]:
            print("  {}: {}".format(stem, "; ".join(errs[:3])))
    if inst and min(inst[i] for i in range(nc)) < 0.1 * max(inst.values()):
        print("\n  >> Class imbalance > 10x. Collect more data for the minority class(es).")

    # ---------------------------------------------------------------- grouping
    rx = re.compile(a.group_pattern) if a.group_pattern else None
    groups = defaultdict(list)
    for s in images:
        key = original_stem(s)
        if rx:
            m = rx.match(key)
            key = m.group(1) if m else key
        groups[key].append(s)
    print("\n{} images form {} independent groups".format(len(images), len(groups)))
    if len(groups) < 0.2 * len(images):
        print("  (mostly video frames - good that they are grouped)")

    if a.dry_run:
        return

    # ---------------------------------------------------------------- split
    split = split_groups(groups.keys(), a.split, a.seed)
    out = os.path.abspath(a.out)
    for part, gkeys in split.items():
        img_dir = os.path.join(out, "images", part)
        lbl_dir = os.path.join(out, "labels", part)
        for d in (img_dir, lbl_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        n = 0
        for g in gkeys:
            for stem in groups[g]:
                src = images[stem]
                dst = os.path.join(img_dir, os.path.basename(src))
                if a.symlink:
                    os.symlink(os.path.abspath(src), dst)
                else:
                    shutil.copy2(src, dst)
                with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                    for c, cx, cy, w, h in clean_labels[stem]:
                        f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(c, cx, cy, w, h))
                n += 1
        print("{:<5}: {:>6} images ({} groups)".format(part, n, len(gkeys)))

    cfg = {"path": out, "train": "images/train", "val": "images/val", "test": "images/test",
           "names": {i: n for i, n in enumerate(names)}}
    yaml_path = os.path.join(os.path.dirname(out), "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write("# Generated by data/prepare_dataset.py - class ids MUST match the label files\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("\nWrote {}".format(yaml_path))

    stats = {
        "classes": [{"id": i, "name": n, "boxes": inst[i], "images": img_per_cls[i]} for i, n in enumerate(names)],
        "images": len(images), "groups": len(groups), "background_images": background,
        "splits": {p: sum(len(groups[g]) for g in k) for p, k in split.items()},
        "box_sizes_at_imgsz": {"imgsz": a.imgsz, **{n: sizes[n] for n, _ in SIZE_BINS}},
        "grouped_by": a.group_pattern,
    }
    stats_path = os.path.join(os.path.dirname(out), "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("Wrote {} (used by docs/update_readme.py)".format(stats_path))


if __name__ == "__main__":
    main()
