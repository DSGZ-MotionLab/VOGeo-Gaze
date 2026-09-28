"""Video inference.

One network pass over each recording in clips, then one eyeball centre for the
recording (runtime/center.py) and a re-lift of every frame onto it; the overlay is
drawn afterwards from the final records.  Several videos can share a forward pass
(``batch``); decoding and encoding run in threads beside the GPU.
"""
import json
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from tqdm import tqdm

from vogeo_gaze.runtime.center import fit_recording_center
from vogeo_gaze.runtime.overlay import blend_segmentation, draw_segmentation, render_video
from vogeo_gaze.runtime.records import KEYS, concat, write_csv
from vogeo_gaze.runtime.videoio import FrameSource, VideoSink, probe_fps

CLIP_FRAMES = 16        # frames per clip seen by F_tem
REFINER_HZ = 12.0       # rate F_tem was trained at
MAX_STRIDE = 16


@dataclass
class Options:
    source_fpx: float | None = None  # native focal length of the camera (px), if known
    batch: int = 1                  # videos per forward pass
    start: int = 0
    frames: int | None = None
    video: bool = True              # overlay.mp4
    mesh: bool = True               # eyeball and cornea wireframes in the overlay
    side_by_side: bool = True       # input frame beside the overlay
    seg_video: bool = False         # segmentation.mp4
    autocast: bool = False          # fp16 on CUDA


def stride_for(path):
    """Source frames per F_tem frame."""
    fps = probe_fps(path)
    return max(1, min(int(round(fps / REFINER_HZ)), MAX_STRIDE)) if fps > 0 else 1


class _Stream:
    """One video in flight; its record chunks (and segmentation frames) are
    collected in order in a worker thread."""

    def __init__(self, path, out_dir, source, seg_video, img_size):
        self.path, self.out_dir, self.source, self.chunks = path, out_dir, source, []
        self.seg_writer = (VideoSink(os.path.join(out_dir, "segmentation.mp4"), source.fps, img_size[::-1])
                           if seg_video else None)
        self._q, self._error = queue.Queue(maxsize=4), None
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        while (item := self._q.get()) is not None:
            chunk, seg = item
            try:
                self.chunks.append(chunk)
                if self.seg_writer is not None:
                    self.seg_writer.write(draw_segmentation(seg, chunk))
            except BaseException as exc:
                self._error = exc
                return

    def submit(self, chunk, seg):
        if self._error is not None:
            raise self._error
        self._q.put((chunk, seg))

    def close(self):
        self._q.put(None)
        self._thread.join()
        if self.seg_writer is not None:
            self.seg_writer.close()
        if self._error is not None:
            raise self._error


class Engine:
    def __init__(self, model, opts: Options | None = None):
        self.model, self.opts = model, opts or Options()
        self.device = next(model.parameters()).device

    def run(self, jobs):
        """``jobs``: (video path, output directory) pairs.  Videos whose frame rates
        need different F_tem strides run in separate groups."""
        groups = {}
        for path, out_dir in jobs:
            os.makedirs(out_dir, exist_ok=True)
            groups.setdefault(stride_for(path), []).append((path, out_dir))
        for stride, group in groups.items():
            self._run_group(group, stride)

    def _open(self, path, out_dir, clip_len):
        opts = self.opts
        end = None if opts.frames is None else opts.start + opts.frames - 1
        source = FrameSource(path, clip_len, self.model.img_size, start=opts.start, end=end,
                             source_fpx=opts.source_fpx, model_fpx=self.model.fpx)
        if not opts.source_fpx:
            print(f"[camera] {os.path.basename(path)}: {source.src_w}x{source.src_h}, assumed native "
                  f"focal length {source.implied_fpx:.1f} px (set --source-fpx if known)")
        return _Stream(path, out_dir, source, opts.seg_video, self.model.img_size)

    def _forward(self, gray, stride):
        x = torch.from_numpy(gray).to(self.device, non_blocking=True)
        frames = x.to(torch.float32).mul_(1.0 / 255.0).unsqueeze(2).expand(-1, -1, 3, -1, -1)
        amp = torch.autocast("cuda", dtype=torch.float16, enabled=self.opts.autocast and self.device.type == "cuda")
        with torch.no_grad(), amp:
            out = self.model(frames, stride=stride)
        seg = None
        if self.opts.seg_video:
            B, T = gray.shape[:2]
            seg = blend_segmentation(out["seg_logits"].flatten(0, 1), x.flatten(0, 1)).reshape(
                B, T, *gray.shape[2:], 3).cpu().numpy()
        return {k: out[k].float().cpu().numpy() for k in KEYS}, seg

    def _run_group(self, jobs, stride):
        opts, clip_len = self.opts, CLIP_FRAMES * stride
        if stride > 1:
            print(f"[rate] clips of {clip_len} frames, every {stride}th feeds F_tem (~{REFINER_HZ:g} Hz)")
        pending, live, done = list(jobs), [], []
        while pending and len(live) < opts.batch:
            live.append(self._open(*pending.pop(0), clip_len))
        known = sum(s.source.total_frames or 0 for s in live)
        bar = tqdm(total=known if not pending and known else None, desc="network", unit="frame")
        t0, n_frames = time.perf_counter(), 0
        while live:
            rows = []
            for s in list(live):
                clip = s.source.next_clip()
                if clip is None:
                    s.source.close()
                    live.remove(s)
                    done.append(s)
                    if pending:
                        live.append(self._open(*pending.pop(0), clip_len))
                else:
                    rows.append((s, *clip))
            if not rows:
                continue
            out, seg = self._forward(np.stack([r[1] for r in rows]), stride)
            for i, (s, gray, n, first) in enumerate(rows):
                chunk = {k: v[i, :n] for k, v in out.items()}
                chunk["frame"] = np.arange(first, first + n)
                s.submit(chunk, None if seg is None else seg[i, :n])
                n_frames += n
                bar.update(n)
        bar.close()
        for s in done:
            s.close()
        seconds = time.perf_counter() - t0
        runtime = {"frames": n_frames, "seconds": round(seconds, 3), "fps": round(n_frames / max(seconds, 1e-9), 1)}
        print(f"[vogeo_gaze] {n_frames} frames in {seconds:.2f} s = {runtime['fps']} fps")
        for s in done:
            self._finish(s, stride, runtime)

    def _finish(self, s, stride, runtime):
        opts, fps = self.opts, s.source.fps
        if not s.chunks:
            print(f"[skip] no frames read from {s.path}")
            return
        rec = concat(s.chunks)
        info = {"video": os.path.basename(s.path), "fps": fps, "refiner_stride": stride,
                "options": asdict(opts), "runtime": runtime}
        try:
            info["eyeball_center"] = fit_recording_center(rec)
        except ValueError as exc:
            info["eyeball_center"] = {"error": f"{exc}; per-clip centres kept"}
        write_csv(rec, fps, os.path.join(s.out_dir, "gaze.csv"))
        if opts.video:
            n, secs = render_video(s.path, rec, os.path.join(s.out_dir, "overlay.mp4"), self.model.fpx,
                                   self.model.img_size, CLIP_FRAMES * stride, opts.mesh, opts.side_by_side,
                                   opts.start, opts.source_fpx)
            print(f"[overlay] {n} frames in {secs:.2f} s = {n / max(secs, 1e-9):.1f} fps")
        with open(os.path.join(s.out_dir, "run_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"[done] {s.out_dir}")
