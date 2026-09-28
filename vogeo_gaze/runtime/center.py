"""One eyeball centre per recording, and the re-lift of every frame onto it.

F_tem estimates one centre per clip (paper, Eq. 4).  Assuming the eye does not move
relative to the camera during a recording, their median is used as the eyeball
centre of the whole recording, and every frame's corrected pupil-centre ray is
lifted onto the sphere of radius L_p about it.
"""
import numpy as np

from vogeo_gaze.geometry.anatomy import R_EYE, R_IRIS


def relift(pupil, gaze, center, l_p):
    """Each pupil-centre ray onto the sphere (center, l_p); a miss keeps its gaze."""
    pupil, gaze = np.asarray(pupil, np.float64), np.asarray(gaze, np.float64)
    center = np.broadcast_to(np.asarray(center, np.float64), pupil.shape)
    ray = pupil / np.maximum(np.linalg.norm(pupil, axis=1, keepdims=True), 1e-12)
    b = np.sum(ray * center, axis=1)
    disc = b * b - (np.sum(center * center, axis=1) - float(l_p) ** 2)
    hit = np.isfinite(ray).all(axis=1) & np.isfinite(disc) & (disc > 1e-8) & (b > 0)
    lifted = (b - np.sqrt(np.maximum(disc, 0.0)))[:, None] * ray - center
    lifted /= np.maximum(np.linalg.norm(lifted, axis=1, keepdims=True), 1e-12)
    gaze = gaze.copy()
    gaze[hit] = lifted[hit]
    gaze /= np.maximum(np.linalg.norm(gaze, axis=1, keepdims=True), 1e-12)
    return gaze, center + float(l_p) * gaze, hit


def fit_recording_center(rec):
    """Replace the per-clip centres in ``rec`` (whole-recording arrays) in place; return a report."""
    centers = np.asarray(rec["c_eye"], np.float64)
    centers = centers[np.isfinite(centers).all(axis=1)]
    if len(centers) < 8:
        raise ValueError("too few finite eye centres")
    center = np.median(centers, axis=0)
    gaze, pupil, hit = relift(rec["c_pupil"], rec["gaze"], center, np.sqrt(R_EYE ** 2 - R_IRIS ** 2))
    n = len(gaze)
    rec["c_eye"] = np.broadcast_to(np.asarray(center, np.float32), (n, 3)).copy()
    rec["c_pupil"], rec["gaze"] = pupil.astype(np.float32), gaze.astype(np.float32)
    rec["gaze_ok"] = hit.astype(np.float32)
    rec["gaze_angles"] = np.stack([np.degrees(np.arctan2(gaze[:, 0], -gaze[:, 2])),
                                   np.degrees(np.arcsin(np.clip(gaze[:, 1], -1, 1)))], axis=1).astype(np.float32)
    return {"center_mm": np.round(center, 4).tolist(), "ray_hit_fraction": round(float(hit.sum()) / max(n, 1), 4)}
