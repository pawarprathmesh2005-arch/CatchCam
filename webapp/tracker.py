"""
Person tracking for footage uploaded through the web interface.

Writes outputs/<camera_id>_observations.json in the SAME format the rest of CatchCam reads:
  {"camera_id", "track_id", "frame", "timestamp", "class", "confidence", "bbox_xyxy"}
timestamp = time_offset + frame / fps  (the same clock as track.py / vehicle_intel, so cameras line up).

Uses YOLO + ByteTrack from ultralytics (already a CatchCam dependency).
"""
import json
from pathlib import Path

import cv2


def run(video: str, camera_id: str, out_dir: str, weights: str = "yolov8n.pt", every: int = 1, conf: float = 0.35,
        time_offset: float = 0.0, progress=None, imgsz: int = None, skip_empty: bool = False, notes: list = None) -> str:
    """every=N really skips frames (the detector never sees them), so every=3 is about 3x faster. imgsz shrinks the picture for detection.
    skip_empty=True first finds the stretches where something moves and only runs the detector on those (see webapp/motion.py)."""
    from ultralytics import YOLO            # lazy: only needed when footage is actually processed
    every = max(1, int(every))
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    # plan = [(first_frame, last_frame, source_file_or_None)]; None means "the whole video in one go"
    plan = [(0, total, None)]
    if skip_empty:
        from webapp.motion import motion_segments
        segs, n_frames, fps = motion_segments(video, min_blob=0.0005 if conf < 0.3 else 0.0012)
        covered = sum(b - a for a, b in segs)
        if n_frames and covered < 0.8 * n_frames:
            plan = [(a, b, "clip") for a, b in segs]
            if notes is not None:
                notes.append(f"Skipped the empty stretches: analysed {covered / fps:.0f}s of {n_frames / fps:.0f}s ({covered / n_frames:.0%}), "
                             f"the parts where something moves." if segs else
                             f"No movement found in {n_frames / fps:.0f}s of video, so nobody was detected. If people are in it, "
                             f"process again with 'Analyse everything' and Sensitivity High.")
    to_do = max(sum(b - a for a, b, _ in plan), 1)

    try:                                   # leave one CPU core free so the web page and search stay responsive while this runs
        import os
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    except Exception:                      # noqa: BLE001  torch missing or odd build: not important
        pass
    model = YOLO(weights)
    extra = {"imgsz": int(imgsz)} if imgsz else {}
    rows, done, next_id, tmp = [], 0, 1, None
    if any(src for _, _, src in plan):
        import tempfile
        tmp = tempfile.TemporaryDirectory(prefix="catchcam_clips_")
    try:
        for seg_no, (start, end, src) in enumerate(plan):
            if src:
                from webapp.motion import write_clip
                src = str(Path(tmp.name) / f"seg{seg_no}.avi")
                write_clip(video, start, end, src)
            stream = model.track(source=src or video, stream=True, persist=not src, tracker="bytetrack.yaml", conf=conf,
                                 classes=[0], verbose=False, vid_stride=every, **extra)   # class 0 = person
            ids = {}                                                          # this stretch's tracker ids -> unique ids for the camera
            for n, r in enumerate(stream):
                i = n * every                                                 # frame number inside this stretch
                if progress and n % 25 == 0:
                    progress(min((done + i) / to_do, 1.0))
                if r.boxes is None or r.boxes.id is None:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy().tolist()
                tids = r.boxes.id.int().tolist()
                confs = r.boxes.conf.cpu().numpy().tolist()
                for box, tid, c in zip(xyxy, tids, confs):
                    if src:                                                   # separate stretches: number the people 1, 2, 3 ... without clashes
                        if tid not in ids:
                            ids[tid] = next_id
                            next_id += 1
                        out_id = ids[tid]
                    else:
                        out_id = int(tid)
                    frame = start + i
                    rows.append({"camera_id": camera_id, "track_id": out_id, "frame": frame,
                                 "timestamp": round(time_offset + frame / fps, 3), "class": "person",
                                 "confidence": round(float(c), 3), "bbox_xyxy": [int(round(float(v))) for v in box]})      # whole pixels, like track.py: extract_embeddings.py slices frames with these
            done += end - start
            if src:
                Path(src).unlink(missing_ok=True)
    finally:
        if tmp:
            tmp.cleanup()
    out = Path(out_dir) / f"{camera_id}_observations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    if progress:
        progress(1.0)
    return str(out)
