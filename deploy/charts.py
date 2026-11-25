"""Plotly figures for the Streamlit dashboard (pure functions -> testable without Streamlit).

Colour rules: categorical slots in a fixed order, one palette per theme (dark steps are chosen
for the dark surface, not inverted), text in ink colours rather than series colours, thin bars
with rounded data ends and 2 px surface-coloured gaps between stacked segments.
"""
import plotly.graph_objects as go

THEMES = {
    "light": {"surface": "#ffffff", "ink": "#1f1f1e", "muted": "#6b6a64", "grid": "#e6e5e0",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]},
    "dark": {"surface": "#0e1117", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#34342f",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c"]},
}
STAGES = [("pre", "preprocess"), ("infer", "inference"), ("post", "decode + NMS"), ("track", "ByteTrack")]


def _base(fig, t, height, title=None):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=76 if title else 36, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["ink"], size=12),
        title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top", yref="container",
                   font=dict(size=14)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="normal",
                    font=dict(color=t["muted"])),
        hoverlabel=dict(bgcolor=t["surface"], font_color=t["ink"], bordercolor=t["grid"]),
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=t["grid"], zeroline=False, linecolor=t["grid"], tickfont=dict(color=t["muted"]))
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False, linecolor=t["grid"], tickfont=dict(color=t["muted"]))
    return fig


def class_counts(now, unique, names, theme="light", height=260, max_classes=8):
    """Grouped bars per class: objects in the current frame vs unique tracked objects so far."""
    t = THEMES[theme]
    series = [("in frame now", dict(now))]
    if unique is not None:
        series.append(("unique so far", dict(unique)))
    if len(names) <= max_classes:          # small custom model: always show every class
        classes = [names[k] for k in sorted(names)]
    else:                                  # e.g. 80 COCO classes: only those seen, top N, rest -> "other"
        seen = {}
        for _, c in series:
            for k, v in c.items():
                seen[k] = max(seen.get(k, 0), v)
        classes = sorted(seen, key=lambda k: -seen[k])
        if len(classes) > max_classes:
            rest = classes[max_classes - 1:]
            classes = classes[:max_classes - 1] + ["other"]
            for _, c in series:
                c["other"] = sum(c.pop(k, 0) for k in rest)
    fig = go.Figure()
    if not classes:
        fig.add_annotation(text="no objects yet", showarrow=False, font=dict(color=t["muted"]))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return _base(fig, t, height)
    for i, (label, counts) in enumerate(series):
        vals = [counts.get(c, 0) for c in classes]
        fig.add_bar(x=classes, y=vals, name=label, marker=dict(color=t["series"][i], cornerradius=4),
                    text=vals, textposition="outside", textfont=dict(color=t["ink"]), cliponaxis=False,
                    hovertemplate="%{x}: %{y} " + label + "<extra></extra>")
    fig.update_layout(barmode="group", bargroupgap=0.12, showlegend=len(series) > 1)
    fig.update_yaxes(rangemode="tozero", dtick=max(1, _nice_step(max([0] + [v for _, c in series for v in c.values()]))))
    return _base(fig, t, height)


def _nice_step(vmax):
    for s in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if vmax / s <= 6:
            return s
    return 1000


def latency_gauge(ms, budget_ms=33.3, theme="light", height=190, title="Frame latency"):
    """Bullet gauge: current end-to-end latency against the real-time budget (30 FPS = 33.3 ms)."""
    t = THEMES[theme]
    top = max(budget_ms * 2, ms * 1.15)
    fig = go.Figure(go.Indicator(
        mode="number+gauge", value=ms, number=dict(suffix=" ms", font=dict(color=t["ink"], size=24)),
        gauge=dict(
            shape="bullet", axis=dict(range=[0, top], tickcolor=t["muted"], tickfont=dict(color=t["muted"])),
            bar=dict(color=t["series"][0], thickness=0.45),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, budget_ms], color=t["grid"])],
            threshold=dict(line=dict(color=t["ink"], width=2), thickness=0.9, value=budget_ms)),
        domain=dict(x=[0.24, 0.76], y=[0.15, 0.85]),  # leaves room for the number on the right
        title=dict(text="{}<br><span style='font-size:11px'>| = {:.0f} FPS budget</span>".format(
            title, 1000.0 / budget_ms), font=dict(size=13, color=t["muted"]))))
    return _base(go.Figure(fig), t, height)


def latency_timeline(history, budget_ms=33.3, theme="light", height=230):
    """history: list of dicts with 'infer' and 'total' ms (most recent last)."""
    t = THEMES[theme]
    x = list(range(-len(history) + 1, 1))
    fig = go.Figure()
    for i, (key, label) in enumerate((("total", "detect + track"), ("infer", "inference only"))):
        fig.add_scatter(x=x, y=[h[key] for h in history], name=label, mode="lines",
                        line=dict(color=t["series"][i], width=2),
                        hovertemplate=label + ": %{y:.1f} ms<extra></extra>")
    fig.add_hline(y=budget_ms, line=dict(color=t["muted"], width=1, dash="dot"),
                  annotation_text="{:.0f} FPS budget".format(1000.0 / budget_ms),
                  annotation_font=dict(color=t["muted"], size=11), annotation_position="top left")
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title=dict(text="frames ago", font=dict(color=t["muted"], size=11)))
    fig.update_yaxes(title=dict(text="ms", font=dict(color=t["muted"], size=11)), rangemode="tozero")
    return _base(fig, t, height)


def backend_breakdown(rows, theme="light", height=None, title="Latency per frame, by stage"):
    """Horizontal stacked bars: pre / infer / post / track ms per backend, total + FPS labelled."""
    t = THEMES[theme]
    labels = ["{}<br><span style='font-size:11px'>{}</span>".format(r["model"], r["backend"]) for r in rows][::-1]
    fig = go.Figure()
    for i, (key, name) in enumerate(STAGES):
        vals = [r.get(key + "_ms", 0.0) for r in rows][::-1]
        if not any(vals):
            continue
        fig.add_bar(y=labels, x=vals, name=name, orientation="h",
                    marker=dict(color=t["series"][i], line=dict(color=t["surface"], width=2)),
                    hovertemplate=name + ": %{x:.2f} ms<extra></extra>")
    totals = [r["total_ms"] for r in rows][::-1]
    fig.add_scatter(y=labels, x=totals, mode="text", showlegend=False, hoverinfo="skip",
                    text=["  {:.1f} ms | {:.0f} FPS".format(r["total_ms"], r["fps"]) for r in rows][::-1],
                    textposition="middle right", textfont=dict(color=t["ink"], size=12))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title=dict(text="ms per frame (lower is better)", font=dict(color=t["muted"], size=11)),
                     range=[0, max(totals) * 1.45 if totals else 1])
    return _base(fig, t, height or 110 + 62 * len(rows), title)


def device_comparison(df, theme="light", height=320):
    """Saved benchmark rows (benchmark/results/speed.csv): FPS per model, grouped by device."""
    t = THEMES[theme]
    fig = go.Figure()
    for i, (dev, g) in enumerate(df.groupby("device", sort=False)):
        g = g.drop_duplicates("model", keep="last")
        fig.add_bar(x=g["model"] + " (" + g["precision"] + ")", y=g["fps"], name=dev,
                    marker=dict(color=t["series"][i % len(t["series"])], cornerradius=4),
                    hovertemplate="%{x}<br>%{y:.1f} FPS on " + dev + "<extra></extra>")
    fig.add_hline(y=30, line=dict(color=t["muted"], width=1, dash="dot"), annotation_text="30 FPS",
                  annotation_font=dict(color=t["muted"], size=11), annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=dict(text="FPS (higher is better)", font=dict(color=t["muted"], size=11)))
    return _base(fig, t, height, "Saved benchmark results")
