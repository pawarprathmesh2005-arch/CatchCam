"""
Demo data: three short synthetic camera videos with people walking through them, plus the tracking, matching,
clothing and relationship files CatchCam would normally produce. Lets you try the whole interface without any
real footage (and lets the tests exercise it).

    python -m webapp.demo_data demo_data          # then:  set CATCHCAM_DATA_DIR=demo_data   and start the API
    python -m webapp.demo_data demo_data --webm   # same, but WebM videos (play in every browser without conversion)

The story: Asha (green jacket, dark trousers) is seen on cam1, cam2, then reappears on cam3 22 seconds later (a gap).
Ravi (red jacket, blue jeans) walks beside her on cam1 and is also seen on cam2. Meena (blue shirt, black pants) is only on cam3.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, DURATION = 640, 360, 8, 60
BGR = {"green": (60, 170, 60), "red": (50, 50, 200), "blue": (200, 90, 40), "black": (25, 25, 25), "dark": (40, 40, 45),
       "jeans": (170, 80, 40), "skin": (140, 170, 210)}

# camera -> list of (track_id, name, upper, lower, t_start, t_end, lane_y_bottom, x_from, x_to)
PEOPLE = {
    "cam1": [(1, "Asha", "green", "dark", 2, 14, 300, -60, 700), (2, "Ravi", "red", "jeans", 4, 16, 322, -90, 680)],
    "cam2": [(1, "Asha", "green", "dark", 18, 30, 310, 700, -60), (2, "Ravi", "red", "jeans", 26, 38, 300, 700, -80)],
    "cam3": [(1, "Asha", "green", "dark", 52, 59, 305, -60, 700), (2, "Meena", "blue", "black", 10, 28, 315, 700, -60)],
}
MATCHES = [("cam1", 1, "cam2", 1, 0.91), ("cam2", 1, "cam3", 1, 0.84), ("cam1", 2, "cam2", 2, 0.88)]


def _person_box(cx, bottom):
    h = 128
    w = 52
    return [cx - w / 2, bottom - h, cx + w / 2, bottom]


def _draw_background(frame, cam_index):
    frame[:] = (62 + 8 * cam_index, 58, 52)
    cv2.rectangle(frame, (0, 0), (W, 120), (84, 78, 70), -1)                 # wall
    cv2.rectangle(frame, (0, 120), (W, H), (96 + 6 * cam_index, 100, 104), -1)   # floor
    for x in range(0, W, 80):
        cv2.line(frame, (x, 120), (x - 40 * (cam_index + 1), H), (110, 112, 116), 1)
    cv2.rectangle(frame, (40 + cam_index * 120, 30), (150 + cam_index * 120, 110), (140, 150, 120), 2)   # a door / window


def _draw_person(frame, box, upper, lower, t):
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    cv2.circle(frame, ((x1 + x2) // 2, y1 + int(h * 0.09)), int(h * 0.075), BGR["skin"], -1)
    cv2.rectangle(frame, (x1, y1 + int(h * 0.18)), (x2, y1 + int(h * 0.58)), BGR[upper], -1)
    cv2.rectangle(frame, (x1 + 4, y1 + int(h * 0.58)), (x2 - 4, y2), BGR[lower], -1)
    swing = int(6 * np.sin(t * 6))
    cv2.line(frame, ((x1 + x2) // 2 - 8, y2 - 6), ((x1 + x2) // 2 - 8 + swing, y2), (20, 20, 20), 4)


def build(out_dir, webm=False):
    out = Path(out_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    (out / "matches").mkdir(exist_ok=True)
    (out / "relationships").mkdir(exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    videos = {}
    ext, fourcc = (".webm", "VP80") if webm else (".mp4", "mp4v")
    for ci, (cam, people) in enumerate(PEOPLE.items()):
        vpath = out / "videos" / f"{cam}{ext}"
        vw = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*fourcc), FPS, (W, H))
        obs = []
        frame = np.zeros((H, W, 3), np.uint8)
        for i in range(FPS * DURATION):
            t = i / FPS
            _draw_background(frame, ci)
            for (tid, name, up, lo, t0, t1, bottom, xa, xb) in people:
                if t0 <= t <= t1:
                    cx = xa + (xb - xa) * (t - t0) / (t1 - t0)
                    box = _person_box(cx, bottom + 3 * np.sin(t * 5))
                    if box[2] < 0 or box[0] > W:
                        continue
                    _draw_person(frame, box, up, lo, t)
                    clipped = [max(box[0], 0), box[1], min(box[2], W - 1), box[3]]
                    obs.append({"camera_id": cam, "track_id": tid, "frame": i, "timestamp": round(t, 3), "class": "person",
                                "confidence": 0.9, "bbox_xyxy": [round(v, 1) for v in clipped]})
            cv2.putText(frame, f"{cam}  {int(t // 60):02d}:{int(t % 60):02d}", (14, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
            vw.write(frame)
        vw.release()
        (out / "outputs" / f"{cam}_observations.json").write_text(json.dumps(obs))
        videos[cam] = vpath.relative_to(out).as_posix()
    (out / "camera_videos.json").write_text(json.dumps(videos, indent=2))
    (out / "matches" / "cross_camera_matches.json").write_text(json.dumps(
        [{"camera_a": a, "track_a": ta, "camera_b": b, "track_b": tb, "similarity": s} for a, ta, b, tb, s in MATCHES], indent=2))
    (out / "relationships" / "relationships.json").write_text(json.dumps([
        {"camera_id": "cam1", "track_a": 1, "track_b": 2, "start_timestamp": 4.0, "end_timestamp": 14.0, "duration_seconds": 10.0}], indent=2))

    # clothing colours: the real extractor, run on the synthetic footage
    from person_attrs.extract import extract
    for cam in PEOPLE:
        extract(out / videos[cam], out / "outputs" / f"{cam}_observations.json", cam, out_dir=str(out / "outputs"))
    return out


if __name__ == "__main__":
    target = next((a for a in sys.argv[1:] if not a.startswith("--")), "demo_data")
    build(target, webm="--webm" in sys.argv)
    print(f"Demo data written to {target}/ . Start the API with:  set CATCHCAM_DATA_DIR={target}   (Windows)   "
          f"or  CATCHCAM_DATA_DIR={target} uvicorn api:app")
