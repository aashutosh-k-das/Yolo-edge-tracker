"""Real-time detection + ByteTrack tracking on a video file, webcam, CSI camera or RTSP stream.

Examples (run from the project root):
    # ONNX model, video file -> annotated video + MOT-format track file
    python -m deploy.run --weights weights/yolo_custom.onnx --source demo.mp4 \
        --save out/demo_tracked.mp4 --save-txt out/demo_tracks.txt --reid

    # webcam with a live window (press q / Esc to quit)
    python -m deploy.run --weights weights/yolo_custom.onnx --source 0 --show --reid

    # Jetson Nano: TensorRT engine + CSI camera, headless, watch at http://<jetson-ip>:8080
    python3 -m deploy.run --weights weights/yolo_custom_fp16.engine --source csi --reid --stream-port 8080

Output paths may contain strftime patterns: --save-txt logs/tracks_%Y%m%d_%H%M%S.txt

What you see in the output video:
    coloured box  "#7 drone 0.83"  = confirmed track, id 7 (colour is fixed per id) + motion trail
    grey dashed   "#7 lost"        = occluded track at its Kalman-predicted position; it keeps
                                     id 7 if it re-appears within --track-buffer frames
"""
import argparse
import os
import signal
import time

import cv2

from .detectors import load_detector
from .pipeline import TrackingPipeline
from .camera import ThreadedCamera, is_live, open_source

ORT_PROVIDERS = {"cpu": ["CPUExecutionProvider"],
                 "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                 "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help=".onnx / .engine / .pt")
    p.add_argument("--source", default="0", help="video path, webcam index, 'csi' / 'csi:1', or RTSP URL")
    p.add_argument("--width", type=int, default=1280, help="camera capture width")
    p.add_argument("--height", type=int, default=720, help="camera capture height")
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--gst", action="store_true", help="RTSP via GStreamer hardware H.264 decoding (Jetson)")
    p.add_argument("--no-thread", action="store_true",
                   help="don't use the latest-frame camera thread for live sources")
    p.add_argument("--imgsz", type=int, default=640, help=".pt only; ONNX/TRT use their fixed size")
    p.add_argument("--conf", type=float, default=0.25, help="display threshold when --no-track")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--names", default=None, help="dataset YAML or 'aircraft,drone,vehicle' (default: data/dataset.yaml)")
    p.add_argument("--ort-provider", choices=sorted(ORT_PROVIDERS), default=None,
                   help="ONNX Runtime device (default: best available)")
    g = p.add_argument_group("tracking")
    g.add_argument("--no-track", action="store_true", help="detection only")
    g.add_argument("--track-thresh", type=float, default=0.5, help="high-score threshold (stage 1)")
    g.add_argument("--low-thresh", type=float, default=0.1, help="low-score threshold (stage 2)")
    g.add_argument("--match-thresh", type=float, default=0.8)
    g.add_argument("--track-buffer", type=int, default=30,
                   help="frames an occluded track is kept (30 = 1 s @30fps; try 60-90 for long occlusions)")
    g.add_argument("--reid", action="store_true",
                   help="re-identify lost tracks by motion + colour (hovering / fast drones)")
    g.add_argument("--hide-lost", action="store_true", help="don't draw occluded tracks")
    o = p.add_argument_group("output")
    o.add_argument("--show", action="store_true", help="open a display window")
    o.add_argument("--save", default=None, help="annotated video path (.mp4)")
    o.add_argument("--save-txt", default=None,
                   help="MOT-challenge track file: frame,id,x,y,w,h,score,cls,-1,-1 (for TrackEval)")
    o.add_argument("--stream-port", type=int, default=0, help="serve annotated MJPEG on http://<host>:PORT")
    o.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def _sigterm(signum, frame):
    # systemd stops services with SIGTERM; turn it into an exception so `finally` closes the video file
    raise KeyboardInterrupt


def main():
    a = parse_args()
    signal.signal(signal.SIGTERM, _sigterm)
    a.save = time.strftime(a.save) if a.save else None
    a.save_txt = time.strftime(a.save_txt) if a.save_txt else None
    kw = dict(imgsz=a.imgsz, conf=a.conf, iou=a.iou, names=a.names)
    if a.ort_provider and a.weights.endswith(".onnx"):
        kw["providers"] = ORT_PROVIDERS[a.ort_provider]
    det = load_detector(a.weights, **kw)
    cap = open_source(a.source, a.width, a.height, a.cam_fps, gst=a.gst)
    live = is_live(a.source)
    if live and not a.no_thread:
        cap = ThreadedCamera(cap)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    fps_in = fps_in if 1 < fps_in < 240 else 30
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print("Model: {} ({}) | source: {} @ {:.0f} fps{}".format(
        a.weights, det.backend, a.source, fps_in, " | {} frames".format(n_total) if n_total > 0 else ""))

    pipe = TrackingPipeline(det, track=not a.no_track, track_thresh=a.track_thresh,
                            low_thresh=a.low_thresh, match_thresh=a.match_thresh,
                            track_buffer=a.track_buffer, frame_rate=fps_in, display_conf=a.conf,
                            reid=a.reid, show_lost=not a.hide_lost)
    det.warmup(5)

    server = None
    if a.stream_port:
        from .stream import MJPEGServer
        server = MJPEGServer(a.stream_port)
        print("Streaming on http://0.0.0.0:{} (open http://<this-device-ip>:{} in a browser)".format(
            a.stream_port, a.stream_port))
    draw = a.show or a.save is not None or server is not None

    writer, txt = None, None
    if a.save_txt:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_txt)), exist_ok=True)
        txt = open(a.save_txt, "w", buffering=1)  # line-buffered: survives power loss better
    t_start = time.time()
    info = {}
    lifetimes = {}  # track id -> number of frames it was visible
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out, info = pipe.process(frame, draw=draw)
            if server:
                server.update(out)

            if pipe.tracker is not None:
                for x1, y1, x2, y2, tid, score, cls in info["objects"]:
                    lifetimes[int(tid)] = lifetimes.get(int(tid), 0) + 1
                    if txt:
                        txt.write("{},{},{:.1f},{:.1f},{:.1f},{:.1f},{:.3f},{},-1,-1\n".format(
                            pipe.frames, int(tid), x1, y1, x2 - x1, y2 - y1, score, int(cls)))
            if a.save:
                if writer is None:
                    os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
                writer.write(out)
            if a.show:
                cv2.imshow("yolo-edge-tracker", out)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if pipe.frames % 50 == 0:
                print("frame {:5d}{} | {:5.1f} FPS | infer {:5.1f} ms | track {:4.1f} ms | {} tracks, {} occluded".format(
                    pipe.frames, "/{}".format(n_total) if n_total > 0 else "", info["fps"], info["infer"],
                    info["track_ms"], info["num_objects"], info.get("num_lost", 0)))
            if a.max_frames and pipe.frames >= a.max_frames:
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        if writer:
            writer.release()
        if txt:
            txt.close()
        if a.show:
            cv2.destroyAllWindows()
        if server:
            server.close()
        det.close()

    elapsed = time.time() - t_start
    print("\nProcessed {} frames in {:.1f}s -> {:.1f} FPS average".format(
        pipe.frames, elapsed, pipe.frames / max(elapsed, 1e-6)))
    if isinstance(cap, ThreadedCamera) and cap.dropped:
        print("Camera frames skipped to stay real-time: {} (inference slower than the camera)".format(cap.dropped))
    if pipe.tracker is not None:
        short = sum(1 for n in lifetimes.values() if n < 10)
        print("Tracks: {} unique ids | per class {} | re-identifications {} | short-lived (<10 frames) {}".format(
            len(lifetimes), info.get("counts", {}), info.get("reid_events", 0), short))
        if lifetimes and short > 0.3 * len(lifetimes):
            print("  Many short tracks -> likely ID switches or flickering detections: try --reid, a larger "
                  "--track-buffer, or a lower --track-thresh.")
    if a.save:
        print("Annotated video ->", a.save)
    if a.save_txt:
        print("MOT track file  ->", a.save_txt)


if __name__ == "__main__":
    main()
