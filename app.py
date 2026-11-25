"""Phase 7 - Streamlit dashboard.

    streamlit run app.py

Tab "Live tracking"      upload a video / use a camera / single image -> detection + ByteTrack IDs,
                         FPS + latency meter, per-class counts (Plotly), annotated video download.
Tab "Backend comparison" PyTorch vs ONNX Runtime vs TensorRT on the same frames: stage breakdown,
                         speed-up, detection agreement, side-by-side annotated frames, saved results
                         from other devices (e.g. the Jetson's benchmark/results/speed.csv).
"""
import glob
import os
import tempfile
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from benchmark.benchmark import bench, load_frames
from deploy import charts
from deploy.camera import ThreadedCamera, is_live, open_source
from deploy.detectors import load_detector
from deploy.pipeline import TrackingPipeline
from deploy.utils import draw_detections

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES = os.path.join(ROOT, "data", "dataset", "images", "test")
SPEED_CSV = os.path.join(ROOT, "benchmark", "results", "speed.csv")
st.set_page_config(page_title="Edge Object Tracker", layout="wide")


# ======================================================================= helpers
def ui_theme():
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


@st.cache_resource(show_spinner="Loading model...")
def get_detector(path, imgsz):
    return load_detector(path, imgsz=imgsz)


def find_weights():
    pats = ["weights/*.pt", "weights/*.onnx", "weights/*.engine", "runs/*/*/weights/best.pt"]
    return sorted(set(p for pat in pats for p in glob.glob(os.path.join(ROOT, pat))))


BACKEND_ORDER = {".pt": 0, ".onnx": 1, ".engine": 2, ".trt": 2, ".plan": 2}


def by_backend(paths):
    """PyTorch first (the baseline), then ONNX, then TensorRT."""
    return sorted(paths, key=lambda p: (BACKEND_ORDER.get(os.path.splitext(p.split("@")[0])[1], 9), p))


def rel(p):
    return os.path.relpath(p, ROOT) if os.path.isabs(p) else p


def open_writer(stem, fps, size):
    """Prefer browser-playable codecs (H.264 mp4, then VP8 webm); fall back to mp4v (download only)."""
    for fourcc, ext, playable in (("avc1", ".mp4", True), ("VP80", ".webm", True), ("mp4v", ".mp4", False)):
        w = cv2.VideoWriter(stem + ext, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, stem + ext, playable
        w.release()
    raise IOError("No usable video codec found in this OpenCV build")


def class_counts_now(objects, names):
    counts = {}
    for row in objects:
        n = names.get(int(row[-1]), str(int(row[-1])))
        counts[n] = counts.get(n, 0) + 1
    return counts


def chart(slot, fig, key):
    slot.plotly_chart(fig, key=key, config={"displayModeBar": False})


# ======================================================================= sidebar
TH = ui_theme()
st.sidebar.title("Settings")
options = find_weights()
choice = st.sidebar.selectbox("Model", options + ["Custom path..."], format_func=rel)
weights = st.sidebar.text_input("Weights path", "weights/model.onnx") if choice == "Custom path..." else choice
imgsz = st.sidebar.select_slider("Input size (.pt only)", [320, 416, 512, 640, 960, 1280], value=640,
                                 help=".onnx / .engine models have a fixed input size")

st.sidebar.subheader("Tracking")
use_track = st.sidebar.toggle("ByteTrack", value=True)
use_reid = st.sidebar.toggle("Re-identification", value=True, disabled=not use_track,
                             help="Re-link lost tracks by motion + colour (hovering / fast drones)")
show_lost = st.sidebar.toggle("Show occluded tracks", value=True, disabled=not use_track)
conf = st.sidebar.slider("Display confidence (no tracking)", 0.05, 0.95, 0.25, 0.05)
track_thresh = st.sidebar.slider("High-score threshold", 0.2, 0.9, 0.5, 0.05, disabled=not use_track)
match_thresh = st.sidebar.slider("Match threshold", 0.5, 0.95, 0.8, 0.05, disabled=not use_track)
track_buffer = st.sidebar.slider("Lost-track buffer (frames)", 5, 120, 30, 5, disabled=not use_track)

st.sidebar.subheader("Display")
target_fps = st.sidebar.number_input("Real-time target (FPS)", 5, 120, 30, help="sets the latency budget line")
display_every = st.sidebar.number_input("Show every Nth frame", 1, 10, 1,
                                        help="Drawing to the browser costs time; higher = closer to true model FPS")
chart_every = st.sidebar.slider("Chart refresh (s)", 0.2, 3.0, 0.5, 0.1)
BUDGET = 1000.0 / target_fps

# ======================================================================= header
st.title("Real-Time Object Detection & Tracking on Edge")
st.caption("YOLOv8 · ONNX / TensorRT · ByteTrack · classes from data/dataset.yaml")

detector = None
if weights and (os.path.exists(weights.split("@")[0]) or weights.endswith(".pt")):
    try:
        detector = get_detector(weights, imgsz)
        st.sidebar.success("{} | {} px | {} classes".format(detector.backend, detector.imgsz, len(detector.names)))
    except Exception as e:
        st.sidebar.error("Could not load {}: {}".format(rel(weights), e))
else:
    st.sidebar.info("Put a trained model in `weights/` (Phase 2/3) or enter a path.")

tab_live, tab_cmp = st.tabs(["Live tracking", "Backend comparison"])


# ======================================================================= live tab
def make_pipeline(fps):
    detector.conf = conf
    detector.iou = 0.45
    return TrackingPipeline(detector, track=use_track, track_thresh=track_thresh, match_thresh=match_thresh,
                            track_buffer=track_buffer, frame_rate=fps, display_conf=conf,
                            reid=use_reid, show_lost=show_lost)


def run_stream(cap, fps, record=True):
    pipe = make_pipeline(fps)
    names = detector.names
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    st.button("Stop", key="stop_btn", help="Stops processing; the video recorded so far stays downloadable")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        frame_slot = st.empty()
        progress = st.progress(0.0, text="starting...") if total > 0 else None
        timeline_slot = st.empty()
    with right:
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        s_fps, s_proc, s_inf, s_trk = m1.empty(), m2.empty(), m3.empty(), m4.empty()
        gauge_slot = st.empty()
        counts_slot = st.empty()

    history = deque(maxlen=150)
    writer = None
    last_chart, n_chart = 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, info = pipe.process(frame)
            proc_ms = info["detect_ms"] + info["track_ms"]
            history.append({"infer": info["infer"], "total": proc_ms})

            if record:
                if writer is None:
                    stem = os.path.join(tempfile.gettempdir(), "tracked_{}".format(int(time.time())))
                    writer, out_path, playable = open_writer(stem, fps, (out.shape[1], out.shape[0]))
                    st.session_state["last_output"] = {"path": out_path, "playable": playable}
                writer.write(out)

            if pipe.frames % display_every == 0:
                frame_slot.image(out[:, :, ::-1], channels="RGB", width="stretch")
            now = time.time()
            if now - last_chart >= chart_every:
                last_chart, n_chart = now, n_chart + 1
                s_fps.metric("Pipeline FPS", "{:.1f}".format(info["fps"]), help="including drawing + browser updates")
                s_proc.metric("Processing FPS", "{:.1f}".format(1000.0 / max(proc_ms, 1e-3)),
                              help="1000 / (detect + track ms): what the model + tracker sustain")
                s_inf.metric("Inference", "{:.1f} ms".format(info["infer"]))
                s_trk.metric("Tracks / occluded", "{} / {}".format(info["num_objects"], info["num_lost"]),
                             help="re-identifications so far: {}".format(info["reid_events"]))
                chart(gauge_slot, charts.latency_gauge(proc_ms, BUDGET, TH), "gauge_%d" % n_chart)
                chart(counts_slot, charts.class_counts(
                    class_counts_now(info["objects"], names), info["counts"] if use_track else None,
                    names, TH), "counts_%d" % n_chart)
                chart(timeline_slot, charts.latency_timeline(list(history), BUDGET, TH), "timeline_%d" % n_chart)
                if progress is not None:
                    progress.progress(min(pipe.frames / float(total), 1.0),
                                      text="frame {} / {}".format(pipe.frames, total))
    finally:  # also runs when Stop triggers a rerun, so the partial video is still saved
        cap.release()
        if writer is not None:
            writer.release()
    if progress is not None:
        progress.progress(1.0, text="done: {} frames".format(pipe.frames))


def output_section():
    last = st.session_state.get("last_output")
    if not last or not os.path.exists(last["path"]):
        return
    st.subheader("Annotated output")
    if last["playable"]:
        st.video(last["path"])
    else:
        st.caption("This OpenCV build has no browser codec (H.264/VP8): download to watch.")
    with open(last["path"], "rb") as f:
        st.download_button("Download annotated video", f, file_name="tracked" + os.path.splitext(last["path"])[1],
                           mime="video/webm" if last["path"].endswith(".webm") else "video/mp4", type="primary")


with tab_live:
    if detector is None:
        st.info("Load a model in the sidebar to start.")
    else:
        source = st.radio("Input", ["Upload video", "Camera", "Image"], horizontal=True, label_visibility="collapsed")
        if source == "Upload video":
            up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "webm"])
            if up is not None and st.button("Run detection + tracking", type="primary"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
                tmp.write(up.getvalue())
                tmp.close()
                st.session_state["last_upload"] = tmp.name
                cap = cv2.VideoCapture(tmp.name)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                run_stream(cap, fps if 1 < fps < 240 else 30)
        elif source == "Camera":
            st.caption("Uses the camera attached to the machine running Streamlit (webcam index, 'csi' on a "
                       "Jetson, or an RTSP URL). Press **Stop** to end.")
            c1, c2 = st.columns([3, 1])
            cam = c1.text_input("Camera", "0")
            record = c2.toggle("Record", value=True)
            if st.button("Start camera", type="primary"):
                try:
                    cap = open_source(cam)
                    cap = ThreadedCamera(cap) if is_live(cam) else cap
                    run_stream(cap, 30, record=record)
                except IOError as e:
                    st.error(str(e))
        else:
            up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if up is not None:
                img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                out, info = TrackingPipeline(detector, track=False, display_conf=conf).process(img)
                c1, c2 = st.columns([3, 2])
                c1.image(out[:, :, ::-1], channels="RGB", width="stretch")
                c2.metric("Inference", "{:.1f} ms".format(info["infer"]))
                chart(c2, charts.class_counts(class_counts_now(info["objects"], detector.names), None,
                                              detector.names, TH), "img_counts")
                ok, png = cv2.imencode(".png", out)
                c2.download_button("Download annotated image", png.tobytes(), "detections.png", "image/png")
        output_section()


# ======================================================================= comparison tab
with tab_cmp:
    st.markdown("Runs every selected model on the **same frames** with the **same pre/post-processing**, "
                "so differences come from the inference engine. TensorRT engines only load on the NVIDIA "
                "device they were built on.")
    models = st.multiselect("Models", by_backend(options), default=by_backend(options)[:4], format_func=rel,
                            help="The first model (PyTorch if selected) is the baseline for speed-up and agreement")
    c1, c2, c3, c4 = st.columns(4)
    add_fp16 = c1.toggle("+ PyTorch FP16", value=False, help="needs a CUDA GPU")
    track_in_loop = c2.toggle("Include ByteTrack", value=True)
    iters = c3.select_slider("Timed frames", [30, 50, 100, 200, 500], value=100)
    warmup = c4.select_slider("Warm-up", [5, 10, 20, 50], value=10)

    src_opts = []
    if os.path.isdir(TEST_IMAGES):
        src_opts.append("Test images")
    if st.session_state.get("last_upload"):
        src_opts.append("Last uploaded video")
    src_opts.append("Random frames")
    frame_src = st.radio("Frames", src_opts, horizontal=True,
                         help="Random noise gives few detections, which under-reports NMS + tracking cost")

    if st.button("Run comparison", type="primary", disabled=not models):
        frames = load_frames(images=TEST_IMAGES if frame_src == "Test images" else None,
                             video=st.session_state.get("last_upload") if frame_src == "Last uploaded video" else None,
                             n=60)
        todo = by_backend(list(models) + ([m + "@fp16" for m in models if m.endswith(".pt")] if add_fp16 else []))
        probe = frames[::max(1, len(frames) // 20)][:20]  # spread over the clip for agreement + preview
        rows, samples, skipped = [], [], []
        bar = st.progress(0.0)
        for k, m in enumerate(todo):
            bar.progress(k / float(len(todo)), text="{} ...".format(rel(m)))
            try:
                det = get_detector(m, imgsz)
                cb = lambda f, k=k, m=m: bar.progress((k + f) / float(len(todo)), text="{} ...".format(rel(m)))
                row = bench(m, frames, imgsz, warmup, iters, track_in_loop, det=det, progress=cb)
                det.conf = 0.25
                rows.append(row)
                samples.append((row, [det(f) for f in probe], det.names))
            except Exception as e:
                skipped.append("{}: {}".format(rel(m), e))
        bar.empty()
        if rows:
            from export.verify import match_detections
            base = rows[0]["total_ms"]
            ref_dets = samples[0][1]
            for row, dets, _ in samples:
                row["speedup"] = base / row["total_ms"]
                n_ref = n_match = 0
                for a, b in zip(ref_dets, dets):
                    nr, _, ious, _ = match_detections(a, b)
                    n_ref, n_match = n_ref + nr, n_match + len(ious)
                row["agreement"] = n_match / float(n_ref) if n_ref else float("nan")
        # preview the frame where the baseline found the most objects
        best = max(range(len(probe)), key=lambda i: len(samples[0][1][i])) if samples else 0
        st.session_state["cmp"] = {"rows": rows, "skipped": skipped, "frame": probe[best],
                                   "samples": [(r["model"], r["backend"], d[best], n) for r, d, n in samples]}

    res = st.session_state.get("cmp")
    if res:
        for s in res["skipped"]:
            st.warning("Skipped " + s)
        rows = res["rows"]
        if rows:
            cols = st.columns(len(rows))
            for i, (col, r) in enumerate(zip(cols, rows)):
                col.metric("{} · {}".format(r["model"], r["precision"]), "{:.0f} FPS".format(r["fps"]),
                           delta="{:+.0f}% vs {} ({:.2f}x)".format((r["speedup"] - 1) * 100, rows[0]["model"],
                                                                  r["speedup"]) if i else "baseline",
                           delta_color="normal" if i else "off")
            chart(st, charts.backend_breakdown(rows, TH), "breakdown")
            table = pd.DataFrame(rows)[["model", "backend", "precision", "size_mb", "pre_ms", "infer_ms", "post_ms",
                                        "track_ms", "total_ms", "p95_ms", "fps", "speedup", "agreement"]]
            st.dataframe(table, hide_index=True, column_config={
                "speedup": st.column_config.NumberColumn(format="%.2fx"),
                "agreement": st.column_config.ProgressColumn(
                    "agreement", help="share of the baseline's boxes found again (same class, IoU >= 0.5)",
                    min_value=0.0, max_value=1.0, format="percent")})

            st.subheader("Same frame, side by side")
            cols = st.columns(len(res["samples"]))
            for col, (model, backend, dets, names) in zip(cols, res["samples"]):
                img = draw_detections(res["frame"].copy(), dets, names)
                col.image(img[:, :, ::-1], channels="RGB", width="stretch",
                          caption="{} · {} · {} boxes".format(model, backend, len(dets)))

    st.divider()
    if os.path.exists(SPEED_CSV):
        saved = pd.read_csv(SPEED_CSV)
        devices = list(saved["device"].unique())
        pick = st.multiselect("Devices in benchmark/results/speed.csv", devices, default=devices)
        sub = saved[saved["device"].isin(pick)]
        if len(sub):
            chart(st, charts.device_comparison(sub, TH), "devices")
            with st.expander("Raw rows"):
                st.dataframe(sub, hide_index=True)
    else:
        st.caption("Tip: run `python3 -m benchmark.benchmark ...` on the Jetson and copy "
                   "benchmark/results/speed.csv here to compare devices in this dashboard.")
