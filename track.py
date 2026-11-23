"""
CatchCam — Week 1: Single-Camera Detection + Tracking
=======================================================
Runs YOLO detection + ByteTrack tracking on a single camera's video feed.
Outputs:
  1. An annotated video with bounding boxes + persistent track IDs
  2. A JSON file of structured track observations — this is the schema
     that feeds Week 2 (Re-ID / cross-camera matching) and Week 3
     (Space-Time Graph), so keep it as-is.

Usage:
    python catchcam_track.py --source path/to/video.mp4 --camera_id cam1

Requires: ultralytics, opencv-python (already in your venv)
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def run_tracking(source: str, camera_id: str, model_name: str = "yolov8n.pt",
                  conf: float = 0.35, classes: list[int] | None = None,
                  out_dir: str = "outputs", time_offset: float = 0.0):
    """
    Detect + track objects in a single video stream.

    classes: COCO class ids to keep. None = all classes.
             Common ones: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck.
             For a security/surveillance use case, [0] (person only) or
             [0, 2, 3, 5, 7] (people + vehicles) are typical choices.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)  # auto-downloads weights on first run

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    video_out_file = out_path / f"{camera_id}_tracked.mp4"
    writer = cv2.VideoWriter(
        str(video_out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    observations = []
    frame_idx = 0
    wall_clock_start = time_offset

    results_stream = model.track(
        source=source,
        conf=conf,
        classes=classes,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in results_stream:
        frame = result.orig_img.copy()
        timestamp = wall_clock_start + (frame_idx / fps)

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, tid, cf, cid in zip(boxes, track_ids, confs, cls_ids):
                x1, y1, x2, y2 = box.astype(int)
                label = f"ID {tid} | {model.names[cid]} | {cf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                observations.append({
                    "camera_id": camera_id,
                    "frame": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "track_id": int(tid),
                    "class": model.names[cid],
                    "confidence": round(float(cf), 4),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                })

        writer.write(frame)
        frame_idx += 1

    writer.release()

    json_out_file = out_path / f"{camera_id}_observations.json"
    with open(json_out_file, "w") as f:
        json.dump(observations, f, indent=2)

    print(f"[CatchCam] Done. {frame_idx} frames processed.")
    print(f"[CatchCam] Annotated video -> {video_out_file}")
    print(f"[CatchCam] Track observations -> {json_out_file}")
    print(f"[CatchCam] Unique track IDs seen: {len(set(o['track_id'] for o in observations))}")

    return observations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatchCam Week 1: single-camera YOLO + ByteTrack")
    parser.add_argument("--source", required=True)
    parser.add_argument("--camera_id", default="cam1")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--classes", nargs="*", type=int, default=None)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--time_offset", type=float, default=0.0)
    args = parser.parse_args()

    run_tracking(
        source=args.source,
        camera_id=args.camera_id,
        model_name=args.model,
        conf=args.conf,
        classes=args.classes,
        out_dir=args.out_dir,
        time_offset=args.time_offset,
    )