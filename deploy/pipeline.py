"""Detector + ByteTrack + visualisation, shared by the CLI (deploy/run.py) and Streamlit (app.py)."""
import time
from collections import defaultdict, deque

from track import BYTETracker, tracks_to_array

from .utils import draw_detections, draw_hud, draw_lost, draw_tracks


class TrackingPipeline(object):
    def __init__(self, detector, track=True, track_thresh=0.5, low_thresh=0.1,
                 match_thresh=0.8, track_buffer=30, frame_rate=30, trail_len=30,
                 display_conf=0.25, reid=False, show_lost=True):
        self.detector = detector
        self.names = detector.names
        self.track = track
        self.display_conf = display_conf
        if track:
            # ByteTrack needs the low-score boxes, so the detector threshold must sit at low_thresh.
            detector.conf = min(detector.conf, low_thresh)
            self.tracker = BYTETracker(track_thresh, low_thresh, match_thresh,
                                       track_buffer, frame_rate, reid=reid)
        else:
            detector.conf = display_conf
            self.tracker = None
        self.trail_len = trail_len
        self.reid = reid
        self.show_lost = show_lost
        self.reset()

    def reset(self):
        if self.tracker:
            self.tracker.reset()
        self.trails = defaultdict(lambda: deque(maxlen=self.trail_len))
        self.track_cls = {}                  # track id -> current (majority-vote) class, for counting
        self.fps = 0.0
        self.frames = 0
        self._last = None

    def process(self, frame, draw=True):
        """Returns (annotated_frame, info dict)."""
        t0 = time.perf_counter()
        dets = self.detector(frame)
        t1 = time.perf_counter()

        lost = []
        if self.tracker:
            tracks = tracks_to_array(self.tracker.update(dets, frame if self.reid else None))
            lost = tracks_to_array(self.tracker.lost_tracks())
            for x1, y1, x2, y2, tid, _, cls in tracks:
                self.trails[int(tid)].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                self.track_cls[int(tid)] = int(cls)
            objects = tracks
        else:
            objects = dets
        t2 = time.perf_counter()

        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last = now
        self.frames += 1

        info = dict(self.detector.timings)
        info.update({
            "detect_ms": (t1 - t0) * 1e3,
            "track_ms": (t2 - t1) * 1e3,
            "fps": self.fps,
            "num_dets": int(len(dets)),
            "num_objects": int(len(objects)),
            "num_lost": int(len(lost)),
            "reid_events": self.tracker.reid_events if self.tracker else 0,
            "counts": self._unique_counts(),
        })

        out = frame
        if draw:
            out = frame.copy()
            if self.tracker:
                if self.show_lost and len(lost):
                    draw_lost(out, lost)
                draw_tracks(out, objects, self.names, self.trails)
            else:
                draw_detections(out, objects, self.names)
            draw_hud(out, [
                "{} | {:.1f} FPS".format(self.detector.backend, self.fps),
                "infer {:.1f} ms | track {:.1f} ms".format(info["infer"], info["track_ms"]),
                "tracks: {} | occluded: {} | re-ID: {}".format(
                    len(objects), info["num_lost"], info["reid_events"]) if self.tracker
                else "detections: {}".format(len(objects)),
            ])
        info["objects"] = objects  # (N, 7) tracks or (N, 6) detections
        return out, info

    def _unique_counts(self):
        """Unique tracked objects per class; each track counted once, under its majority class."""
        counts = defaultdict(int)
        for c in self.track_cls.values():
            counts[self.names.get(c, str(c))] += 1
        return dict(counts)
