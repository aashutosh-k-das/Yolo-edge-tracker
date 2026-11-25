"""Fill the README's dataset + benchmark tables from your REAL result files (no hand-typed numbers).

    python docs/update_readme.py

Reads (whichever exist):
    data/dataset_stats.json            written by data/prepare_dataset.py
    benchmark/results/speed.csv        written by benchmark/benchmark.py (one block per device)
    benchmark/results/accuracy.csv     written by benchmark/eval_accuracy.py
    benchmark/results/power.json       optional: {"<device>": {"<model>": watts}} from parse_tegrastats
and rewrites the sections between <!-- NAME:START --> / <!-- NAME:END --> markers in README.md.
Also lists README images under docs/assets/ that do not exist yet.
"""
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
RES = os.path.join(ROOT, "benchmark", "results")


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(":---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def dataset_section():
    path = os.path.join(ROOT, "data", "dataset_stats.json")
    if not os.path.exists(path):
        return None
    s = json.load(open(path))
    cls = pd.DataFrame(s["classes"])[["name", "boxes", "images"]]
    cls.columns = ["Class", "Boxes", "Images"]
    sp = s["splits"]
    sizes = s["box_sizes_at_imgsz"]
    total_boxes = max(sum(sizes[k] for k in sizes if k != "imgsz"), 1)
    small = sizes.get("tiny <16px", 0) + sizes.get("small <32px", 0)
    out = [md_table(cls), "",
           "**{:,} images** ({:,} background / negative) · split train / val / test = **{:,} / {:,} / {:,}**{}.".format(
               s["images"], s["background_images"], sp.get("train", 0), sp.get("val", 0), sp.get("test", 0),
               " (grouped by source video, {} groups, so no frame leakage between splits)".format(s["groups"])
               if s.get("grouped_by") else ""),
           "**{:.0f}%** of boxes are smaller than 32 px at {} px input (typical for distant drones).".format(
               100.0 * small / total_boxes, sizes["imgsz"])]
    return "\n".join(out)


def benchmark_section():
    sp_path = os.path.join(RES, "speed.csv")
    if not os.path.exists(sp_path):
        return None
    speed = pd.read_csv(sp_path).drop_duplicates(["device", "model"], keep="last")
    acc = pd.read_csv(os.path.join(RES, "accuracy.csv")) if os.path.exists(os.path.join(RES, "accuracy.csv")) else None
    power = json.load(open(os.path.join(RES, "power.json"))) if os.path.exists(os.path.join(RES, "power.json")) else {}
    blocks = []
    for device, g in speed.groupby("device", sort=False):
        g = g.copy()
        base = g["total_ms"].iloc[0]
        t = pd.DataFrame({
            "Model": g["model"], "Backend": g["backend"], "Precision": g["precision"].str.upper(),
            "Size (MB)": g["size_mb"], "Inference (ms)": g["infer_ms"].round(1),
            "End-to-end (ms)": g["total_ms"].round(1), "p95 (ms)": g["p95_ms"].round(1),
            "FPS": g["fps"].round(1), "Speed-up": (base / g["total_ms"]).map("{:.2f}x".format)})
        if acc is not None:
            m = g[["model"]].merge(acc, on="model", how="left")
            t["mAP@50"] = m["mAP50"].map(lambda v: "-" if pd.isna(v) else "{:.3f}".format(v)).values
            t["mAP@50-95"] = m["mAP50_95"].map(lambda v: "-" if pd.isna(v) else "{:.3f}".format(v)).values
        if device in power:
            t["FPS/W"] = [("{:.1f}".format(f / power[device][mo]) if mo in power[device] else "-")
                          for mo, f in zip(g["model"], g["fps"])]
        track = " (incl. ByteTrack)" if g["tracking"].astype(str).str.lower().eq("true").all() else ""
        blocks.append("**{}**{}: speed-up relative to `{}`\n\n{}".format(device, track, g["model"].iloc[0], md_table(t)))
    return "\n\n".join(blocks)


def replace(text, name, body):
    pat = re.compile(r"(<!-- {0}:START -->\n).*?(\n<!-- {0}:END -->)".format(name), re.S)
    if not pat.search(text):
        print("marker {} not found in README".format(name))
        return text
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main():
    text = open(README).read()
    for name, fn in (("DATASET", dataset_section), ("BENCHMARK", benchmark_section)):
        body = fn()
        if body:
            text = replace(text, name, body)
            print("updated", name)
        else:
            print("skipped {} (no result files yet)".format(name))
    open(README, "w").write(text)

    missing = sorted(set(p for p in re.findall(r"(docs/assets/[\w./-]+)", text)
                         if not os.path.exists(os.path.join(ROOT, p))))
    if missing:
        print("\nREADME images not created yet (run docs/make_assets.py):")
        for p in missing:
            print("  ", p)


if __name__ == "__main__":
    main()
