"""Per-frame model outputs -> ``gaze.csv``."""
import numpy as np
import pandas as pd

# Model outputs copied to the host for every frame.
KEYS = ("gaze_angles", "gaze", "c_eye", "c_pupil", "r_pupil", "pupil_conf", "iris_conf",
        "open_eye_score", "entpup_el", "iris_el")
BLINK_OPEN_EYE_SCORE = 0.70


def concat(chunks):
    """Clip chunks -> whole-recording arrays, with the source frame index in ``frame``."""
    rec = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    rec["frame"] = rec["frame"].astype(np.int64)
    return rec


def blink_index(closed):
    """Running id of each run of closed-eye frames; 0 where the eye is open."""
    starts = closed & np.concatenate(([True], ~closed[:-1]))
    return np.where(closed, np.cumsum(starts, dtype=np.int32), 0)


def write_csv(rec, fps, path):
    frame = rec["frame"]
    open_eye = rec["open_eye_score"].reshape(-1)
    cols = {"frame": frame, "timestamp": (frame / fps).astype(np.float32),
            "theta_h": rec["gaze_angles"][:, 0], "theta_v": rec["gaze_angles"][:, 1]}
    for name, key in (("gaze", "gaze"), ("c_eye", "c_eye"), ("c_pup", "c_pupil")):
        cols.update({f"{name}_{a}": rec[key][:, i] for i, a in enumerate("xyz")})
    cols.update({"r_pup": rec["r_pupil"].reshape(-1),
                 "pupil_conf": rec["pupil_conf"].reshape(-1), "iris_conf": rec["iris_conf"].reshape(-1),
                 "open_eye_score": open_eye, "blink_index": blink_index(open_eye < BLINK_OPEN_EYE_SCORE)})
    for name in ("entpup", "iris"):
        cols.update({f"{name}_{p}": rec[f"{name}_el"][:, i] for i, p in enumerate(("theta", "cx", "cy", "a", "b"))})
    cols["gaze_ok"] = rec.get("gaze_ok", np.ones(len(frame), np.float32))
    pd.DataFrame(cols).to_csv(path, index=False)
