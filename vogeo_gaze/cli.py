"""python -m vogeo_gaze VIDEO [VIDEO ...] [--out-dir DIR]

Writes gaze.csv, overlay.mp4 and run_info.json for every video to <out-dir>/<video name>/;
the default out-dir is vogeo_results/ next to the video.
"""
import argparse
from pathlib import Path

from vogeo_gaze.models.vogeo_gaze import load_model
from vogeo_gaze.runtime.engine import Engine, Options


def main(argv=None):
    p = argparse.ArgumentParser(prog="vogeo_gaze", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path,
                   help="output root; each video gets <out-dir>/<video name>/ "
                        "(default: vogeo_results/ next to the video)")
    p.add_argument("--weights", help="model weights (default: the bundled MICCAI 2026 weights)")
    p.add_argument("--device", help="cuda | cpu (default: cuda if available)")
    p.add_argument("--source-fpx", type=float, help="native focal length of the camera in pixels, if known")
    p.add_argument("--batch", type=int, default=1, help="videos per forward pass")
    p.add_argument("--start", type=int, default=0, help="first frame")
    p.add_argument("--frames", type=int, help="number of frames")
    p.add_argument("--no-video", action="store_true", help="no overlay.mp4")
    p.add_argument("--no-mesh", action="store_true", help="no eyeball/cornea wireframe in the overlay")
    p.add_argument("--overlay-only", action="store_true", help="overlay without the input frame beside it")
    p.add_argument("--seg-video", action="store_true", help="also write segmentation.mp4")
    p.add_argument("--autocast", action="store_true", help="fp16 autocast on CUDA")
    a = p.parse_args(argv)
    for v in a.videos:
        if not v.is_file():
            p.error(f"no such video: {v}")
    opts = Options(source_fpx=a.source_fpx, batch=max(1, a.batch), start=a.start, frames=a.frames,
                   video=not a.no_video, mesh=not a.no_mesh, side_by_side=not a.overlay_only,
                   seg_video=a.seg_video, autocast=a.autocast)
    jobs = [(str(v), str((a.out_dir or v.parent / "vogeo_results") / v.stem)) for v in a.videos]
    Engine(load_model(a.weights, a.device), opts).run(jobs)
    return 0
