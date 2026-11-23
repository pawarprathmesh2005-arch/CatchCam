"""
CatchCam utility: pull a few crop images of a specific track_id from a
camera's video, so you can visually verify whether a matched pair really
looks like the same person.

Usage:
    python verify_match.py --video cameras/cam1_topleft.mp4 \
        --observations outputs/cam1_observations.json --track_id 719 \
        --out_dir verify/cam1_track719
"""
import argparse
import json
from pathlib import Path

import cv2


def verify_match(video_path, observations_path, track_id, out_dir, max_crops=5):
    with open(observations_path) as f:
        observations = json.load(f)

    matching = [o for o in observations if o["track_id"] == track_id]
    matching.sort(key=lambda o: o["frame"])

    if not matching:
        print(f"[CatchCam] No observations found for track_id {track_id}")
        return

    step = max(1, len(matching) // max_crops)
    sampled = matching[::step][:max_crops]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    saved = 0
    for i, obs in enumerate(sampled):
        cap.set(cv2.CAP_PROP_POS_FRAMES, obs["frame"])
        ret, frame = cap.read()
        if not ret:
            continue
        x1, y1, x2, y2 = obs["bbox_xyxy"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        # Upscale small crops so they're actually visible when you open them
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        out_file = out_path / f"frame{obs['frame']}_conf{obs['confidence']:.2f}.png"
        cv2.imwrite(str(out_file), crop)
        saved += 1

    cap.release()
    print(f"[CatchCam] Saved {saved} crop images to {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--track_id", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_crops", type=int, default=5)
    args = parser.parse_args()

    verify_match(args.video, args.observations, args.track_id, args.out_dir, args.max_crops)
