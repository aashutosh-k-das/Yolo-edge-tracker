"""Tiny MJPEG-over-HTTP server: watch a headless Jetson's annotated output from any browser.

    python3 -m deploy.run ... --stream-port 8080     ->  open http://<jetson-ip>:8080

JPEG encoding costs CPU (~5-10 ms for 720p on the Nano's A57 cores), so frames are only
encoded while at least one browser is connected, and are downscaled to `max_width`.
No authentication: only use it on a trusted network (or bind to 127.0.0.1 + SSH tunnel).
Python 3.6 compatible (no ThreadingHTTPServer, which is 3.7+).
"""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

PAGE = (b"<html><head><title>yolo-edge-tracker</title></head>"
        b"<body style='margin:0;background:#111;display:flex;justify-content:center'>"
        b"<img src='/stream' style='max-width:100%;max-height:100vh'></body></html>")


class _Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MJPEGServer(object):
    def __init__(self, port=8080, host="0.0.0.0", quality=70, max_width=960):
        self.quality, self.max_width = quality, max_width
        self.cond = threading.Condition()
        self.jpg, self.seq, self.clients = None, 0, 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the console quiet
                pass

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(PAGE)
                    return
                if self.path != "/stream":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with owner.cond:
                    owner.clients += 1
                last = -1
                try:
                    while True:
                        with owner.cond:
                            owner.cond.wait_for(lambda: owner.seq != last, timeout=5.0)
                            data, last = owner.jpg, owner.seq
                        if data is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with owner.cond:
                        owner.clients -= 1

        self.httpd = _Server((host, port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def update(self, frame):
        if self.clients <= 0:
            return  # nobody watching -> no encoding cost
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / float(w))))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if ok:
            with self.cond:
                self.jpg = buf.tobytes()
                self.seq += 1
                self.cond.notify_all()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
