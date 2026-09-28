"""Threaded video decode (gray, letterboxed to the model canvas) and encode."""
import math
import queue
import threading
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Letterbox:
    proc_w: int
    proc_h: int
    pad: tuple            # left, top, right, bottom; negative values crop
    out_w: int
    out_h: int
    scale: float


def letterbox(src_w, src_h, out_w=320, out_h=240, scale=None):
    """Aspect-preserving resize (even size) and symmetric padding into the canvas.

    ``scale`` defaults to filling the canvas; ``model_fpx / source_fpx`` instead
    presents the video at the model's focal length (a larger frame is centre-cropped).
    """
    scale = min(out_w / src_w, out_h / src_h) if scale is None else float(scale)
    if not (scale > 0.0) or not math.isfinite(scale):
        raise ValueError(f"letterbox scale must be positive and finite, got {scale}")
    w = int(round(src_w * scale)) - (int(round(src_w * scale)) % 2)
    h = int(round(src_h * scale)) - (int(round(src_h * scale)) % 2)
    pw, ph = out_w - w, out_h - h
    return Letterbox(w, h, (pw // 2, ph // 2, pw - pw // 2, ph - ph // 2), out_w, out_h, scale)


def to_canvas(frame, lb):
    """BGR or gray frame -> (out_h, out_w) uint8 gray."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if lb.scale == 1.0 and (lb.proc_w, lb.proc_h) == (lb.out_w, lb.out_h) and gray.shape == (lb.out_h, lb.out_w):
        return gray
    gray = cv2.resize(gray, (lb.proc_w, lb.proc_h),
                      interpolation=cv2.INTER_AREA if lb.scale < 1.0 else cv2.INTER_LINEAR)
    left, top, right, bottom = lb.pad
    gray = gray[max(0, -top):gray.shape[0] - max(0, -bottom), max(0, -left):gray.shape[1] - max(0, -right)]
    if max(0, top) or max(0, bottom) or max(0, left) or max(0, right):
        gray = cv2.copyMakeBorder(gray, max(0, top), max(0, bottom), max(0, left), max(0, right),
                                  cv2.BORDER_CONSTANT, value=0)
    return gray


def probe_fps(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    cap.release()
    return max(fps, 0.0)


class FrameSource:
    """Clips of ``clip_len`` canvas frames, decoded ahead in a thread.

    ``next_clip()`` returns ``(gray (T,H,W) uint8, n_valid, first_frame)`` or None; a
    short last clip is padded by repeating its last frame.
    """

    def __init__(self, path, clip_len, img_size=(240, 320), start=0, end=None,
                 source_fpx=None, model_fpx=None, prefetch=3):
        self.path, self.t = str(path), int(clip_len)
        self.h, self.w = img_size
        self.cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise IOError(f"cannot open video: {self.path}")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.fps = fps if fps > 0 else 30.0
        n_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = float(model_fpx) / float(source_fpx) if source_fpx and model_fpx else None
        self.lb = letterbox(self.src_w, self.src_h, self.w, self.h, scale=scale)
        # The native focal length the model implicitly assumes for this video.
        self.implied_fpx = float(model_fpx) / self.lb.scale if model_fpx else None
        self.start = max(0, int(start or 0))
        self.end = None if end is None else (min(int(end), n_total - 1) if n_total > 0 else int(end))
        self.total_frames = (max(0, self.end - self.start + 1) if self.end is not None
                             else max(0, n_total - self.start) if n_total > 0 else None)
        if self.start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        self._cursor, self._done, self._error = self.start, False, None
        self._q = queue.Queue(maxsize=prefetch)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _read_clip(self):
        if self._done:
            return None
        gray = np.empty((self.t, self.h, self.w), np.uint8)
        first, n = self._cursor, 0
        while n < self.t and not (self.end is not None and self._cursor > self.end):
            ok, frame = self.cap.read()
            if not ok:
                break
            gray[n] = to_canvas(frame, self.lb)
            n, self._cursor = n + 1, self._cursor + 1
        if n < self.t:
            self._done = True
        if n == 0:
            return None
        gray[n:] = gray[n - 1]
        return gray, n, first

    def _pump(self):
        try:
            while True:
                clip = self._read_clip()
                self._q.put(clip)
                if clip is None:
                    return
        except BaseException as exc:
            self._error = exc
            self._q.put(None)

    def next_clip(self):
        clip = self._q.get()
        if self._error is not None:
            raise self._error
        return clip

    def close(self):
        self._done = True
        while self._thread.is_alive():
            try:
                self._q.get(timeout=0.1)
            except queue.Empty:
                pass
        self.cap.release()


class VideoSink:
    """mp4 writer fed from a thread; frames must not be modified after ``write``."""

    def __init__(self, path, fps, size_wh, depth=32):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                      max(1, round(float(fps))), tuple(int(v) for v in size_wh))
        if not self.writer.isOpened():
            raise IOError(f"cannot write video: {path}")
        self._q, self._error = queue.Queue(maxsize=depth), None
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        while (frames := self._q.get()) is not None:
            try:
                for frame in frames:
                    self.writer.write(frame)
            except BaseException as exc:
                self._error = exc

    def write(self, frames):
        if self._error is not None:
            raise self._error
        self._q.put(frames)

    def close(self):
        self._q.put(None)
        self._thread.join()
        self.writer.release()
        if self._error is not None:
            raise self._error
