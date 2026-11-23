#!/usr/bin/env python
"""
Process a folder of clips with the same pipeline as the CatchCam web app, then pack the results into one zip.

Meant for a computer with a GPU (Google Colab's free GPU is about 10-20x faster than a laptop CPU), but it runs anywhere.

    python process_clips.py clips --data demo_data \
        --start cam1=12:38:53 --start cam2=12:39:29 --start cam3=12:42:03 --weights yolov8m.pt --speed normal

  * File names are the camera names: clips/cam1.mp4, clips/cam2.mp4 ...
  * --start CAMERA=HH:MM:SS  is the clock shown on that camera at the first frame of its clip (lines the cameras up in time).
  * Result: demo_data/ (videos, outputs, embeddings, matches, time_offsets.json ...) and demo_data.zip.
    On your PC: unzip it, then   $env:CATCHCAM_DATA_DIR = "D:\\CatchCam\\demo_data"   and start CatchCam as usual.
"""
import argparse
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

from webapp import core, jobs


def clock_to_seconds(text: str) -> float:
    parts = text.strip().split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError(f"'{text}' is not a time. Write it as hh:mm:ss, for example 12:38:53.")
    h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) == 3 else 0)
    if h > 23 or m > 59 or s > 59:
        raise ValueError(f"'{text}' is not a valid time of day.")
    return float(h * 3600 + m * 60 + s)


def parse_starts(items) -> dict:
    out = {}
    for item in items or []:
        cam, _, t = item.partition("=")
        if not t:
            raise ValueError(f"--start needs CAMERA=HH:MM:SS, got '{item}'.")
        out[core.safe_camera_id(cam.strip())] = clock_to_seconds(t)
    return out


def _print_progress(job, stop):
    last = None
    while not stop.wait(10):
        line = f"    {job.stage}: {round(job.percent * 100)}%"
        if line != last:
            print(line, flush=True)
            last = line


def make_zip(data: Path, include_videos: bool = True) -> Path:
    out = data.with_suffix(".zip") if data.suffix == "" else data.parent / (data.name + ".zip")
    skip_dirs = {".thumbs", "__pycache__"} | (set() if include_videos else {"videos"})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(data.rglob("*")):
            rel = f.relative_to(data)
            if f.is_file() and f != out and not (set(rel.parts[:-1]) & skip_dirs):
                z.write(f, Path(data.name) / rel)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", help="folder with the clips, named after the cameras: cam1.mp4, cam2.mp4 ...")
    ap.add_argument("--data", default="demo_data", help="where to put the results (default: demo_data)")
    ap.add_argument("--start", action="append", metavar="CAM=HH:MM:SS", help="clock shown on the camera at the first frame (repeat for each camera)")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO model: yolov8n.pt (fast), yolov8s.pt, yolov8m.pt (best on a GPU)")
    ap.add_argument("--speed", default="normal", choices=list(jobs.SPEEDS), help="normal = every frame (use this on a GPU)")
    ap.add_argument("--sensitivity", default="normal", choices=list(jobs.SENSITIVITY))
    ap.add_argument("--all-frames", action="store_true", help="do not skip the stretches where nothing moves")
    ap.add_argument("--no-videos", action="store_true", help="leave the clips out of the zip (you copy them yourself)")
    a = ap.parse_args(argv)

    clips_dir, data = Path(a.clips), Path(a.data)
    clips = sorted(p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() in core.VIDEO_EXT) if clips_dir.is_dir() else []
    if not clips:
        print(f"No video files found in {clips_dir}. Expected something like {clips_dir / 'cam1.mp4'}.", file=sys.stderr)
        return 2
    starts = parse_starts(a.start)
    cams = {}
    for clip in clips:
        cam = core.safe_camera_id(clip.stem)
        dest = data / "videos" / f"{cam}{clip.suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if clip.resolve() != dest.resolve():
            shutil.copy2(clip, dest)
        core.register_video(data, cam, dest)
        cams[cam] = dest
    unknown = set(starts) - set(cams)
    if unknown:
        print(f"--start was given for {sorted(unknown)}, but there is no clip with that name. Clips found: {sorted(cams)}.", file=sys.stderr)
        return 2
    print(f"Processing {len(cams)} cameras: {', '.join(cams)}  (model {a.weights}, speed {a.speed})", flush=True)

    t_all = time.time()
    for cam in cams:
        job = jobs.Job("process", cam)
        job.state = "running"
        stop = threading.Event()
        threading.Thread(target=_print_progress, args=(job, stop), daemon=True).start()
        t0 = time.time()
        print(f"\n== {cam} ==", flush=True)
        try:
            jobs.process_footage(data, cam, job, a.weights, starts.get(cam, 0.0), a.speed, a.sensitivity, not a.all_frames)
        finally:
            stop.set()
        for n in job.notes:
            print(f"  - {n}")
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
    out = make_zip(data, include_videos=not a.no_videos)
    print(f"\nAll cameras done in {time.time() - t_all:.0f}s. Results: {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
