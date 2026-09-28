"""Overlay videos: the fitted eye drawn from the records, and the segmentation."""
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from vogeo_gaze.geometry.camera import pixel_to_ray, project, refract_to_pupil_plane
from vogeo_gaze.geometry.ellipse import ellipse_points
from vogeo_gaze.geometry.render import render_eye
from vogeo_gaze.runtime.videoio import FrameSource, VideoSink

# BGR
ENTPUP = (255, 255, 255)        # detected entrance pupil
IRIS = (255, 255, 50)           # limbus of the fitted eye (detected iris in the segmentation video)
REFRACTED_PUPIL = (0, 255, 255)  # entrance-pupil rim refracted onto the pupil plane
C_EYE = (0, 0, 255)
C_PUPIL = GAZE = (0, 255, 127)
EYEBALL = (193, 182, 255)
CORNEA = (255, 200, 50)
DIM = 0.30                      # model elements on frames whose pupil ray missed the eyeball


def _ellipse(img, el, color, thickness):
    theta, cx, cy, a, b = el
    cv2.ellipse(img, (int(cx + 0.5), int(cy + 0.5)), (int(a + 0.5), int(b + 0.5)),
                float(np.degrees(theta)), 0, 360, color, thickness, lineType=cv2.LINE_AA)


def _point(img, xy, color, radius, w, h):
    x, y = float(xy[0]), float(xy[1])
    if 0 <= x < w and 0 <= y < h:          # False for NaN
        cv2.circle(img, (int(x + 0.5), int(y + 0.5)), radius, color, -1, lineType=cv2.LINE_AA)


def _label(img, text, color, font_scale, top=True):
    """Text on a black box, top right (or bottom left)."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    h, w = img.shape[:2]
    x0, y = (max(0, w - tw - 2), max(0, 1 + th)) if top else (1, min(h - 2 - base, h - 4))
    cv2.rectangle(img, (x0, y - th - 1), (min(w - 1, x0 + tw + 2), min(h - 1, y + base + 1)), (0, 0, 0), -1)
    cv2.putText(img, text, (x0 + 1, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, lineType=cv2.LINE_AA)


def _mesh_points(xy, h, w):
    """(T, rows, cols, 2) projected wireframe -> int32 points and a keep mask."""
    x, y = xy[..., 0], xy[..., 1]
    keep = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= max(h, w) * 100) & (np.abs(y) <= max(h, w) * 100)
    pts = np.empty(xy.shape, np.int32)
    pts[..., 0] = np.rint(np.clip(np.where(keep, x, 0.0), -w * 10, w * 10))
    pts[..., 1] = np.rint(np.clip(np.where(keep, y, 0.0), -h * 10, h * 10))
    return pts, keep


def _mesh_lines(pts, keep, n):
    """Polylines of one frame's wireframe: rows, then columns."""
    lines = []
    for i in range(min(n, pts.shape[0], pts.shape[1])):
        if keep[i].sum() > 1:
            lines.append(pts[i][keep[i]])
        if keep[:, i].sum() > 1:
            lines.append(pts[:, i][keep[:, i]])
    return lines


def draw(gray, rec, fpx, img_size, mesh=True):
    """Frames (T,H,W) uint8 and their records -> (T,H,W,3) overlays."""
    T, h, w = gray.shape
    gaze = torch.as_tensor(np.asarray(rec["gaze"], np.float32))
    eye = render_eye(torch.as_tensor(np.asarray(rec["c_eye"], np.float32)), gaze, fpx, img_size, mesh)
    # Detected entrance-pupil rim, refracted at the fitted cornea onto the pupil plane.
    ent = torch.as_tensor(np.asarray(rec["entpup_el"], np.float32))
    good = torch.isfinite(ent).all(dim=1)
    rays = pixel_to_ray(ellipse_points(torch.where(good[:, None], ent, torch.zeros_like(ent)), 64), fpx, img_size)
    refracted = project(refract_to_pupil_plane(eye["c_cornea"], eye["r_cornea"], gaze, eye["c_pupil"], rays),
                        fpx, img_size).numpy()
    refracted[~good.numpy()] = np.nan
    meshes = ([(*_mesh_points(eye[k].numpy(), h, w), c) for k, c in (("eyeball_mesh", EYEBALL),
                                                                      ("cornea_mesh", CORNEA))]
              if mesh else [])
    c_eye2d, c_pupil2d, limbus = eye["c_eye2d"].numpy(), eye["c_pupil2d"].numpy(), eye["iris_el"].numpy()
    gaze = gaze.numpy()
    angles = rec["gaze_angles"]
    fallback = rec["gaze_ok"].reshape(-1) < 0.5 if "gaze_ok" in rec else np.zeros(T, bool)
    font = 0.3 if max(h, w) <= 400 else 0.4
    out = np.repeat(gray[..., None], 3, axis=-1)
    for i, img in enumerate(out):
        shade = (lambda c: tuple(int(v * DIM) for v in c)) if fallback[i] else (lambda c: c)
        for pts, keep, color in meshes:
            if lines := _mesh_lines(pts[i], keep[i], 25):
                cv2.polylines(img, lines, False, shade(color), 1, lineType=cv2.LINE_AA)
        _point(img, c_eye2d[i], shade(C_EYE), 3, w, h)
        _point(img, c_pupil2d[i], shade(C_PUPIL), 3, w, h)
        for el, color in ((rec["entpup_el"][i], ENTPUP), (limbus[i], shade(IRIS))):
            if np.isfinite(el).all():
                _ellipse(img, el, color, 2)
        if np.isfinite(refracted[i]).all():
            cv2.polylines(img, [np.rint(np.clip(refracted[i], -w * 10, w * 10)).astype(np.int32)], True,
                          shade(REFRACTED_PUPIL), 2, lineType=cv2.LINE_AA)
        p0, g = c_pupil2d[i].astype(np.float64), gaze[i]
        if np.isfinite(p0).all() and np.isfinite(g[:2]).all():
            p1 = (p0[0] + 50.0 * float(g[0]), p0[1] + 50.0 * float(g[1]))
            cv2.line(img, (int(np.clip(p0[0], -w * 10, w * 10) + 0.5), int(np.clip(p0[1], -h * 10, h * 10) + 0.5)),
                     (int(np.clip(p1[0], -w * 10, w * 10) + 0.5), int(np.clip(p1[1], -h * 10, h * 10) + 0.5)),
                     shade(GAZE), 2, lineType=cv2.LINE_AA)
        th, tv = float(angles[i, 0]), float(angles[i, 1])
        _label(img, f"hor={th:.2f}  ver={tv:.2f}" if np.isfinite([th, tv]).all() else "hor=nan  ver=nan",
               (255, 255, 255), font)
        if fallback[i]:
            _label(img, "FALLBACK: clip gaze", (80, 80, 255), font, top=False)
    return out


def render_video(path, rec, out_path, fpx, img_size, clip_len, mesh=True, side_by_side=True,
                 start=0, source_fpx=None):
    """Decode the video again and draw the final records. Returns (frames, seconds)."""
    t0, n_rec, written = time.perf_counter(), len(rec["gaze"]), 0
    source = FrameSource(path, clip_len, img_size, start=start, source_fpx=source_fpx, model_fpx=fpx)
    sink = VideoSink(out_path, source.fps, (img_size[1] * (2 if side_by_side else 1), img_size[0]))
    try:
        bar = tqdm(total=n_rec, desc="overlay", unit="frame")
        while (clip := source.next_clip()) is not None:
            gray, n, first = clip
            a = first - start
            b = min(a + n, n_rec)
            if a >= n_rec:
                break
            frames = draw(gray[:b - a], {k: v[a:b] for k, v in rec.items()}, fpx, img_size, mesh)
            if side_by_side:
                frames = np.concatenate((np.repeat(gray[:b - a, :, :, None], 3, axis=-1), frames), axis=2)
            sink.write(frames)
            written += b - a
            bar.update(b - a)
        bar.close()
    finally:
        source.close()
        sink.close()
    return written, time.perf_counter() - t0


def blend_segmentation(logits, gray):
    """On the device: (N,3,H,W) logits over (N,H,W) uint8 frames -> (N,H,W,3) uint8 BGR."""
    prob = torch.sigmoid(logits.float().clamp(-30.0, 30.0))
    base = gray.to(torch.float32).unsqueeze(-1).expand(-1, -1, -1, 3)
    overlay = torch.stack((prob[:, 0] * 255.0, prob[:, 2] * 200.0, prob[:, 1] * 255.0), dim=-1)
    return (base * 0.65 + overlay * 0.55).clamp_(0.0, 255.0).to(torch.uint8)


def draw_segmentation(blended, rec):
    """Detected ellipses on the blended segmentation frames."""
    out = blended.copy()
    for i, img in enumerate(out):
        for el, color in ((rec["entpup_el"][i], ENTPUP), (rec["iris_el"][i], IRIS)):
            if np.isfinite(el).all() and el[3] >= 1.0:
                _ellipse(img, el, color, 2)
    return out
