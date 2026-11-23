"""
CatchCam — Real-Time Webcam Tracking
========================================
Same YOLO + ByteTrack pipeline as track.py, but pointed at a live camera
(or any continuous stream) instead of a saved video file, running until you
stop it. Results are written out periodically so a dashboard (or anything
else) can poll them without needing a full websocket/streaming setup.

Usage:
    # Live webcam (0 = default camera):
    python realtime_track.py --source 0 --camera_id cam1

    # Test without a webcam, using a saved video as a stand-in for "live":
    python realtime_track.py --source some_video.mp4 --camera_id cam1

    # Headless (no preview window) -- for running on a machine without a
    # monitor, or during automated testing:
    python realtime_track.py --source 0 --camera_id cam1 --no_display

Controls (when display is on): press 'q' in the video window to stop.

Requires: ultralytics, opencv-python (already in your venv)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def run_realtime_tracking(source, camera_id: str, model_name: str = "yolov8n.pt",
                           conf: float = 0.35, classes: list[int] | None = None,
                           out_dir: str = "outputs", display: bool = True,
                           write_interval_frames: int = 30, imgsz: int = 640):
    """
    write_interval_frames: how often (in frames) to save the current
    observations + a snapshot image to disk. 30 frames at ~30fps = once a
    second -- frequent enough for a dashboard to feel "live" without
    hammering the disk every single frame.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    # source can be an int (webcam index) or a string (file path / stream URL)
    try:
        source = int(source)
    except ValueError:
        pass

    observations = []
    frame_idx = 0
    start_time = time.time()

    json_out_file = out_path / f"{camera_id}_observations_live.json"
    snapshot_file = out_path / f"{camera_id}_latest_frame.jpg"

    print(f"[CatchCam] Starting live tracking on source={source}, camera_id={camera_id}")
    print(f"[CatchCam] Writing snapshots -> {json_out_file} and {snapshot_file}")
    if display:
        print("[CatchCam] Press 'q' in the video window to stop.")
    else:
        print("[CatchCam] Running headless. Press Ctrl+C to stop.")

    results_stream = model.track(
        source=source,
        conf=conf,
        classes=classes,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
        imgsz=imgsz,
    )

    try:
        for result in results_stream:
            frame = result.orig_img.copy()
            fps_estimate = frame_idx / max(time.time() - start_time, 1e-6)
            timestamp = time.time() - start_time

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

            cv2.putText(frame, f"LIVE | {fps_estimate:.1f} fps", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if display:
                cv2.imshow(f"CatchCam Live - {camera_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[CatchCam] 'q' pressed, stopping.")
                    break

            # Periodically persist state so a dashboard/other process can poll it
            if frame_idx % write_interval_frames == 0:
                with open(json_out_file, "w") as f:
                    json.dump(observations, f)
                cv2.imwrite(str(snapshot_file), frame)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[CatchCam] Interrupted, stopping.")

    finally:
        # Final save on exit, however it happened
        with open(json_out_file, "w") as f:
            json.dump(observations, f, indent=2)
        if display:
            cv2.destroyAllWindows()

        print(f"[CatchCam] Done. {frame_idx} frames processed.")
        print(f"[CatchCam] Final observations -> {json_out_file}")
        print(f"[CatchCam] Unique track IDs seen: {len(set(o['track_id'] for o in observations))}")

    return observations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CatchCam: real-time webcam/stream tracking")
    parser.add_argument("--source", required=True,
                         help="Webcam index (e.g. 0) or a video file path / stream URL")
    parser.add_argument("--camera_id", default="cam1")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--classes", nargs="*", type=int, default=None)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--no_display", action="store_true",
                         help="Run without a preview window (for headless machines)")
    parser.add_argument("--write_interval_frames", type=int, default=30,
                         help="How often (in frames) to save progress to disk")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Inference resolution. Lower = faster but less accurate. "
                              "Try 320 on CPU-only laptops for a real speed boost. "
                              "Must be a multiple of 32 (e.g. 320, 416, 480, 640).")
    args = parser.parse_args()

    run_realtime_tracking(
        source=args.source,
        camera_id=args.camera_id,
        model_name=args.model,
        conf=args.conf,
        classes=args.classes,
        out_dir=args.out_dir,
        display=not args.no_display,
        write_interval_frames=args.write_interval_frames,
        imgsz=args.imgsz,
    )
