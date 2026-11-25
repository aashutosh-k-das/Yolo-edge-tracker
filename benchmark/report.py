"""Phase 6c - merge speed + accuracy results into a markdown table and a trade-off plot.

    python -m benchmark.report --device "NVIDIA Jetson Nano"

Outputs benchmark/results/report.md and benchmark/results/tradeoff.png (for your report / slides).
"""
import argparse
import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Categorical slots (identity = model size); marker shape = precision (secondary encoding).
SIZE_COLORS = {"n": "#2a78d6", "s": "#eb6834", "m": "#1baf7a"}
PREC_MARKERS = {"fp32": "o", "fp16": "s", "int8": "^"}
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def model_size(name):
    for s in ("n", "s", "m"):
        if "v8" + s in name.lower():
            return s
    return "n"


def plot(df, path, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for _, r in df.iterrows():
        ax.scatter(r["fps"], r["mAP50_95"], s=70, color=SIZE_COLORS[model_size(r["model"])],
                   marker=PREC_MARKERS.get(r["precision"], "o"), edgecolors="white",
                   linewidths=2, zorder=3)
        ax.annotate("{} ({})".format(r["backend"], r["precision"]), (r["fps"], r["mAP50_95"]),
                    xytext=(7, 4), textcoords="offset points", fontsize=8, color=INK_MUTED)
    ax.axvline(30, color=INK_MUTED, lw=1, ls="--", zorder=1)
    ax.text(30, ax.get_ylim()[0], " 30 FPS real-time", fontsize=8, color=INK_MUTED, va="bottom")

    ax.set_xlabel("Throughput (FPS, detection + tracking)", color=INK)
    ax.set_ylabel("mAP50-95 (test split)", color=INK)
    ax.set_title("Speed vs accuracy on {}".format(device), color=INK, loc="left", fontsize=11)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)

    sizes = sorted(set(model_size(m) for m in df["model"]))
    precs = [p for p in PREC_MARKERS if p in set(df["precision"])]
    handles = [Line2D([], [], marker="o", ls="", color=SIZE_COLORS[s], label="YOLOv8" + s) for s in sizes]
    handles += [Line2D([], [], marker=PREC_MARKERS[p], ls="", color=INK_MUTED, label=p.upper()) for p in precs]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    print("Saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", default=os.path.join(RES, "speed.csv"))
    ap.add_argument("--accuracy", default=os.path.join(RES, "accuracy.csv"))
    ap.add_argument("--device", default=None, help="filter speed rows to this device")
    a = ap.parse_args()

    speed = pd.read_csv(a.speed)
    if a.device:
        speed = speed[speed["device"].str.contains(a.device, case=False)]
    speed = speed.drop_duplicates("model", keep="last")
    acc = pd.read_csv(a.accuracy)
    df = speed.merge(acc, on="model", how="left")

    cols = ["model", "backend", "precision", "size_mb", "infer_ms", "total_ms", "p95_ms", "fps",
            "mAP50", "mAP50_95"]
    try:
        md = df[cols].to_markdown(index=False)  # needs `tabulate`
    except ImportError:
        md = df[cols].to_string(index=False)
    device = a.device or (speed["device"].iloc[0] if len(speed) else "device")
    with open(os.path.join(RES, "report.md"), "w") as f:
        f.write("# Benchmark - {}\n\n{}\n".format(device, md))
    print(md)

    plotted = df.dropna(subset=["mAP50_95"])
    if len(plotted):
        plot(plotted, os.path.join(RES, "tradeoff.png"), device)
    else:
        print("No rows with both speed and accuracy - check that model filenames match.")


if __name__ == "__main__":
    main()
