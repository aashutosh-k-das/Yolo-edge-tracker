"""Low-latency camera input for Jetson (Python 3.6 compatible).

Why a thread? cv2.VideoCapture buffers frames. If inference (say 40 ms) is slower than the
camera (33 ms at 30 fps), the buffer fills up and you end up processing frames that are
seconds old. ThreadedCamera reads continuously in the background and hands out only the
NEWEST frame, so latency stays at ~1 frame and slow inference simply drops frames.
Use it for live sources only (camera / RTSP); for video files every frame should be processed.
"""
import threading
import time

import cv2


def gstreamer_csi(sensor_id=0, width=1280, height=720, fps=30, flip=0):
    """CSI camera (IMX219 / Raspberry Pi Cam v2). nvarguscamerasrc + nvvidconv do debayering,
    scaling and colour conversion on the ISP/GPU instead of the CPU."""
    return ("nvarguscamerasrc sensor-id={} ! video/x-raw(memory:NVMM), width={}, height={}, "
            "framerate={}/1, format=NV12 ! nvvidconv flip-method={} ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1").format(
                sensor_id, width, height, fps, flip)


def gstreamer_rtsp(url, latency=0):
    """RTSP H.264 stream decoded by the Nano's hardware decoder (nvv4l2decoder) - saves a lot of CPU."""
    return ("rtspsrc location={} latency={} ! rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=1 max-buffers=1").format(url, latency)


def is_live(source):
    s = str(source).lower()
    return s.isdigit() or s == "csi" or s.startswith(("rtsp://", "http://", "https://"))


def open_source(source, width=1280, height=720, fps=30, gst=False):
    """'0' (USB/V4L2), 'csi', a video file, or an RTSP/HTTP URL (gst=True -> hardware decode)."""
    source = str(source)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(int(source))
        # MJPG lets USB-2 webcams deliver 720p30 (raw YUYV is usually limited to ~10 fps at 720p)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif source.lower() == "csi" or source.lower().startswith("csi:"):
        sensor = int(source.split(":")[1]) if ":" in source else 0
        cap = cv2.VideoCapture(gstreamer_csi(sensor, width, height, fps), cv2.CAP_GSTREAMER)
    elif gst and source.lower().startswith("rtsp://"):
        cap = cv2.VideoCapture(gstreamer_rtsp(source), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        hint = ""
        if source.lower().startswith("csi"):
            hint = " (is nvargus-daemon running? try: sudo systemctl restart nvargus-daemon)"
        elif source.isdigit():
            hint = " (check `ls /dev/video*` and that your user is in the 'video' group)"
        raise IOError("Could not open video source: {}{}".format(source, hint))
    return cap


class ThreadedCamera(object):
    """Drop-in for cv2.VideoCapture.read()/get()/release() that always returns the newest frame."""

    def __init__(self, cap, name="camera"):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame, self.frame_id, self.read_id = None, 0, 0
        self.running, self.ok = True, True
        self.dropped = 0
        self.thread = threading.Thread(target=self._loop, name=name)
        self.thread.daemon = True
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.cond:
                if not ok:
                    self.ok = False
                    self.cond.notify_all()
                    break
                if self.frame_id > self.read_id:
                    self.dropped += 1  # previous frame was never consumed
                self.frame = frame
                self.frame_id += 1
                self.cond.notify_all()

    def read(self, timeout=5.0):
        """Block until a frame newer than the last one returned is available."""
        deadline = time.time() + timeout
        with self.cond:
            while self.frame_id == self.read_id and self.ok:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False, None  # camera stalled
                self.cond.wait(remaining)
            if self.frame_id == self.read_id:
                return False, None
            self.read_id = self.frame_id
            return True, self.frame

    def get(self, prop):
        return self.cap.get(prop)

    def isOpened(self):
        return self.ok and self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
