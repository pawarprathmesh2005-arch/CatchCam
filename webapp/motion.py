"""
Find the stretches of a video where something moves, so the (slow) person detector only runs on those.

CCTV is mostly empty: in a lobby clip with five visitors, about 80% of the frames show nothing but floor. Skipping them makes
processing several times faster and also removes the main source of false detections (the detector never sees an empty scene).
Cameras that are busy all the time simply get one long stretch, so nothing is lost there.
"""
import cv2
import numpy as np


def motion_segments(video: str, step: int = 3, pad_s: float = 1.5, gap_s: float = 2.0, min_blob: float = 0.0012, progress=None):
    """-> (segments, total_frames, fps). segments = [(start_frame, end_frame)], padded and merged, in video frames.
    A frame 'has motion' when a moving blob covers at least min_blob of the picture (about one person's torso on a wide shot).
    Sudden whole-picture changes (auto-exposure, lights switching) are ignored."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    bg = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=36, detectShadows=False)
    kernel = np.ones((3, 3), np.uint8)
    hits, i = [], 0
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                mask = bg.apply(cv2.resize(frame, (320, 180)))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                if 0 < (mask > 0).mean() < 0.6:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours and max(cv2.contourArea(c) for c in contours) / mask.size >= min_blob:
                        hits.append(i)
        i += 1
        if progress and expected and i % 200 == 0:
            progress(min(i / expected, 1.0))
    cap.release()
    total = i
    pad, gap = int(pad_s * fps), int(gap_s * fps)
    segs = []
    for h in hits:
        a, b = max(h - pad, 0), min(h + pad, total)
        if segs and a <= segs[-1][1] + gap:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    return [(a, b) for a, b in segs], total, fps


def write_clip(video: str, start: int, end: int, out_path: str) -> int:
    """Copy frames [start, end) to a small temporary video file; returns how many were written."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    n = 0
    for _ in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
    writer.release()
    cap.release()
    return n
