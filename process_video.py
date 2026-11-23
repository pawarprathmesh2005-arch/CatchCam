"""
Run vehicle intelligence on a video and write <camera_id>_vehicles.json (same folder as the *_observations.json files).

    python -m vehicle_intel.process_video --video cam1.mp4 --camera CAM_1 --out outputs/ [--weights yolov8n.pt] [--every 3]

Needs: ultralytics (YOLO + ByteTrack), opencv-python, and easyocr OR pytesseract (+ tesseract binary).
"""
import argparse
import json
from pathlib import Path

import cv2

from .core import PlateReader, TrackAggregator, VEHICLE_CLASSES, analyze_vehicle


def run(video: str, camera_id: str, out_dir: str, weights: str = "yolov8n.pt", every: int = 3,
        conf: float = 0.35, ocr: str = "auto", time_offset: float = 0.0) -> str:
    from ultralytics import YOLO            # lazy: only needed here
    reader = PlateReader(ocr)
    model = YOLO(weights)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    aggs = {}
    stream = model.track(source=video, stream=True, persist=True, tracker="bytetrack.yaml", conf=conf,
                         classes=list(VEHICLE_CLASSES), verbose=False)
    for i, r in enumerate(stream):
        if r.boxes is None or r.boxes.id is None or i % every:
            continue
        t, frame = time_offset + i / fps, r.orig_img   # same clock as track.py
        for box, tid, cls in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.id.int().tolist(),
                                 r.boxes.cls.int().tolist()):
            x1, y1, x2, y2 = box
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            if crop.shape[0] < 30 or crop.shape[1] < 30:
                continue
            agg = aggs.setdefault(tid, TrackAggregator(camera_id, tid))
            agg.add(t, analyze_vehicle(crop, VEHICLE_CLASSES.get(cls, "unknown"), reader))
    rows = [a.result() for a in aggs.values()]
    out = Path(out_dir) / f"{camera_id}_vehicles.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--camera", required=True)
    ap.add_argument("--out", default="outputs"); ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--every", type=int, default=3); ap.add_argument("--ocr", default="auto")
    ap.add_argument("--time_offset", type=float, default=0.0, help="same value you gave track.py for this camera")
    a = ap.parse_args()
    print("wrote", run(a.video, a.camera, a.out, a.weights, a.every, ocr=a.ocr, time_offset=a.time_offset))
