"""
Extract clothing attributes for every tracked person in a camera video.

    python -m person_attrs.extract --video cameras/cam1_topleft.mp4 \
        --observations outputs/cam1_observations.json --camera_id cam1 --out_dir outputs

Writes outputs/<camera_id>_attributes.json (one row per person track). Needs only opencv + numpy
(no YOLO: it reuses the boxes already in your *_observations.json).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

from .core import PersonAggregator, analyze_person


def extract(video_path, observations_path, camera_id, out_dir="outputs", max_samples=8, min_height=48):
    observations = json.loads(Path(observations_path).read_text(encoding="utf-8-sig"))
    by_track = defaultdict(list)
    for o in observations:
        if o.get("class", "person") == "person":
            by_track[o["track_id"]].append(o)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    rows = []
    for tid, obs in sorted(by_track.items()):
        obs.sort(key=lambda o: o["frame"])
        agg = PersonAggregator(camera_id, tid)
        agg.first_ts, agg.last_ts = obs[0]["timestamp"], obs[-1]["timestamp"]     # true span, not just sampled frames
        step = max(1, len(obs) // max_samples)
        for o in obs[::step][:max_samples]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, o["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = o["bbox_xyxy"]
            crop = frame[max(y1, 0):min(y2, frame.shape[0]), max(x1, 0):min(x2, frame.shape[1])]
            if crop.shape[0] < min_height:
                continue
            agg.add(o["timestamp"], analyze_person(crop))
        agg.first_ts, agg.last_ts = obs[0]["timestamp"], obs[-1]["timestamp"]
        rows.append(agg.result())
    cap.release()
    out = Path(out_dir) / f"{camera_id}_attributes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"[CatchCam] Clothing attributes for {len(rows)} tracks -> {out}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--observations", required=True)
    ap.add_argument("--camera_id", required=True); ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--max_samples", type=int, default=8)
    a = ap.parse_args()
    extract(a.video, a.observations, a.camera_id, a.out_dir, a.max_samples)
