"""Plot training curves from Ultralytics results.csv - one run, or several runs to compare.

    python train/plot_curves.py runs/train/v8n_640
    python train/plot_curves.py runs/train/v8n_640 runs/train/v8n_960 runs/train/v8n_640_tuned \
        --out benchmark/results/training_comparison.png

Panels: box / cls / DFL loss (train solid, val dashed), learning rate, mAP50 (with the target
line), mAP50-95, precision, recall. Also prints a text diagnosis (best epoch, epoch the target
was reached, overfitting check) that you can quote in the report.
"""
import argparse
import os

import pandas as pd

# Categorical slots, assigned in fixed order to runs (identity, never rank).
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

LOSS_PANELS = [("box loss", "train/box_loss", "val/box_loss"),
               ("cls loss", "train/cls_loss", "val/cls_loss"),
               ("DFL loss", "train/dfl_loss", "val/dfl_loss")]
METRIC_PANELS = [("mAP50", "metrics/mAP50(B)"), ("mAP50-95", "metrics/mAP50-95(B)"),
                 ("precision", "metrics/precision(B)"), ("recall", "metrics/recall(B)")]


def load_run(path):
    csv = path if path.endswith(".csv") else os.path.join(path, "results.csv")
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]  # older Ultralytics pads column names
    name = os.path.basename(os.path.dirname(csv))
    return name, df


def diagnose(name, df, target):
    m = df["metrics/mAP50(B)"]
    best_i = int(m.idxmax())
    lines = ["[{}] {} epochs | best mAP50 {:.3f} @ epoch {} | best mAP50-95 {:.3f}".format(
        name, len(df), m.max(), int(df["epoch"][best_i]), df["metrics/mAP50-95(B)"].max())]
    hit = df[m >= target]
    lines.append("  target {:.2f}: {}".format(
        target, "reached at epoch {}".format(int(hit["epoch"].iloc[0])) if len(hit) else "NOT reached"))

    val = df[[c for _, _, c in LOSS_PANELS if c in df]].sum(axis=1)
    trn = df[[c for _, c, _ in LOSS_PANELS if c in df]].sum(axis=1)
    v_min_i = int(val.idxmin())
    if len(df) - v_min_i > 10 and val.iloc[-1] > 1.05 * val.min() and trn.iloc[-1] < trn.iloc[v_min_i]:
        lines.append("  overfitting: val loss bottomed at epoch {} and rose {:.0%} since, while train loss kept "
                     "falling -> best checkpoint is protected; more data / stronger augmentation would help".format(
                         int(df["epoch"][v_min_i]), val.iloc[-1] / val.min() - 1))
    elif len(df) - best_i <= 5:
        lines.append("  still improving at the last epoch -> train longer (more --epochs)")
    return "\n".join(lines)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot(runs, target, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), dpi=130)
    loss_axes, metric_axes = axes[0], axes[1]

    for k, (name, df) in enumerate(runs):
        c = RUN_COLORS[k % len(RUN_COLORS)]
        ep = df["epoch"]
        for ax, (title, tr, va) in zip(loss_axes[:3], LOSS_PANELS):
            if tr in df:
                ax.plot(ep, df[tr], color=c, lw=1.8)
            if va in df:
                ax.plot(ep, df[va], color=c, lw=1.8, ls="--")
        if "lr/pg0" in df:
            loss_axes[3].plot(ep, df["lr/pg0"], color=c, lw=1.8)
        for ax, (title, col) in zip(metric_axes, METRIC_PANELS):
            ax.plot(ep, df[col], color=c, lw=1.8)
        # mark best mAP50 epoch on the mAP50 panel (direct label)
        i = int(df["metrics/mAP50(B)"].idxmax())
        metric_axes[0].scatter([ep[i]], [df["metrics/mAP50(B)"][i]], s=36, color=c,
                               edgecolors="white", linewidths=2, zorder=4)
        metric_axes[0].annotate("{:.3f}".format(df["metrics/mAP50(B)"][i]), (ep[i], df["metrics/mAP50(B)"][i]),
                                xytext=(4, -12), textcoords="offset points", fontsize=8, color=INK)

    for ax, (title, _, _) in zip(loss_axes[:3], LOSS_PANELS):
        style(ax, title)
    style(loss_axes[3], "learning rate")
    for ax, (title, _) in zip(metric_axes, METRIC_PANELS):
        style(ax, title + " (val)")
        ax.set_xlabel("epoch", fontsize=8, color=INK_MUTED)
    metric_axes[0].axhline(target, color=INK_MUTED, lw=1, ls=":")
    metric_axes[0].text(0, target, " target {:.2f}".format(target), fontsize=8, color=INK_MUTED, va="bottom")

    handles = [Line2D([], [], color=RUN_COLORS[k % len(RUN_COLORS)], lw=2, label=n)
               for k, (n, _) in enumerate(runs)]
    handles += [Line2D([], [], color=INK_MUTED, lw=1.8, label="train"),
                Line2D([], [], color=INK_MUTED, lw=1.8, ls="--", label="val")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    print("Saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs (containing results.csv) or csv paths")
    ap.add_argument("--target", type=float, default=0.65)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = [load_run(r) for r in a.runs]
    for name, df in runs:
        print(diagnose(name, df, a.target))
    first = a.runs[0] if os.path.isdir(a.runs[0]) else os.path.dirname(a.runs[0])
    out = a.out or os.path.join(first, "curves.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plot(runs, a.target, out)


if __name__ == "__main__":
    main()
