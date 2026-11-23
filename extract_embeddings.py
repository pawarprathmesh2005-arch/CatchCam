"""
CatchCam — Week 2: Appearance Embedding Extraction
====================================================
For every tracked person (track_id) in a camera's observation JSON, this
crops their bounding box from a few sampled frames, runs each crop through
a pretrained CNN, and averages the results into one "appearance fingerprint"
(embedding vector) per track.

These embeddings are what Week 2's matching step compares across cameras
to figure out "is this person in cam1 the same as this person in cam2?"

Usage:
    python extract_embeddings.py --video cameras/cam1_topleft.mp4 \
        --observations outputs/cam1_observations.json \
        --camera_id cam1 --out_dir embeddings

Requires: torch, torchvision, opencv-python (already in your venv)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


def build_embedder(device):
    """
    Load a pretrained ResNet18 with its classification head removed, so it
    outputs a 512-dim feature vector instead of a class prediction. This is
    a generic ImageNet feature extractor -- good enough to distinguish
    people by clothing color/texture, which is what we need for a first
    working version. (A dedicated person Re-ID model like OSNet would be
    more accurate -- worth mentioning as a future improvement in your report.)
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # strip classification head -> raw features
    model.eval().to(device)

    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 64)),  # tall aspect ratio, matches person shape
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def extract_embeddings(video_path: str, observations_path: str, camera_id: str,
                        out_dir: str = "embeddings", max_samples_per_track: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = build_embedder(device)

    with open(observations_path) as f:
        observations = json.load(f)

    # Group observations by track_id so we can sample a few crops per person
    by_track = defaultdict(list)
    for obs in observations:
        by_track[obs["track_id"]].append(obs)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    track_embeddings = {}

    for track_id, obs_list in by_track.items():
        # Spread sample frames evenly across the track's lifespan rather than
        # just the first few -- gives a more representative appearance average
        obs_list.sort(key=lambda o: o["frame"])
        step = max(1, len(obs_list) // max_samples_per_track)
        sampled = obs_list[::step][:max_samples_per_track]

        crops = []
        for obs in sampled:
            cap.set(cv2.CAP_PROP_POS_FRAMES, obs["frame"])
            ret, frame = cap.read()
            if not ret:
                continue
            x1, y1, x2, y2 = obs["bbox_xyxy"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(preprocess(crop_rgb))

        if not crops:
            continue

        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            features = model(batch)  # shape: (num_crops, 512)
            avg_embedding = features.mean(dim=0)
            avg_embedding = torch.nn.functional.normalize(avg_embedding, dim=0)

        track_embeddings[track_id] = {
            "embedding": avg_embedding.cpu().numpy().tolist(),
            "num_samples": len(crops),
            "first_seen_frame": obs_list[0]["frame"],
            "last_seen_frame": obs_list[-1]["frame"],
            "first_seen_timestamp": obs_list[0]["timestamp"],
            "last_seen_timestamp": obs_list[-1]["timestamp"],
        }

    cap.release()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{camera_id}_embeddings.json"
    with open(out_file, "w") as f:
        json.dump({"camera_id": camera_id, "tracks": track_embeddings}, f)

    print(f"[CatchCam] Extracted embeddings for {len(track_embeddings)} tracks")
    print(f"[CatchCam] Saved -> {out_file}")
    return track_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--camera_id", required=True)
    parser.add_argument("--out_dir", default="embeddings")
    parser.add_argument("--max_samples_per_track", type=int, default=5)
    args = parser.parse_args()

    extract_embeddings(
        video_path=args.video,
        observations_path=args.observations,
        camera_id=args.camera_id,
        out_dir=args.out_dir,
        max_samples_per_track=args.max_samples_per_track,
    )
