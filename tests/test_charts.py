import pandas as pd
import pytest

charts = pytest.importorskip("deploy.charts")
NAMES = {0: "aircraft", 1: "drone", 2: "vehicle"}
ROWS = [dict(model="m.pt", backend="pytorch-cuda", precision="fp32", pre_ms=4, infer_ms=18, post_ms=2,
             track_ms=0.5, total_ms=24.5, fps=40.8),
        dict(model="m_fp16.engine", backend="tensorrt", precision="fp16", pre_ms=4, infer_ms=3, post_ms=2,
             track_ms=0.5, total_ms=9.5, fps=105.3)]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_figures_build(theme):
    f = charts.class_counts({"drone": 2}, {"drone": 5, "vehicle": 1}, NAMES, theme)
    assert [t.name for t in f.data] == ["in frame now", "unique so far"]
    assert list(f.data[0].y) == [0, 2, 0] and list(f.data[1].y) == [0, 5, 1]
    assert len(charts.class_counts({"drone": 2}, None, NAMES, theme).data) == 1
    assert charts.latency_gauge(20.0, 33.3, theme).data[0].value == 20.0
    hist = [{"infer": 10, "total": 20}] * 5
    assert len(charts.latency_timeline(hist, 33.3, theme).data) == 2
    f = charts.backend_breakdown(ROWS, theme)
    assert [t.name for t in f.data if t.type == "bar"] == ["preprocess", "inference", "decode + NMS", "ByteTrack"]
    df = pd.DataFrame([dict(device="T4", model="m", precision="fp16", fps=100),
                       dict(device="Nano", model="m", precision="fp16", fps=24)])
    assert [t.name for t in charts.device_comparison(df, theme).data] == ["T4", "Nano"]
