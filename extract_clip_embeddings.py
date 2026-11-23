"""
CatchCam — CLIP Embedding Extraction (for natural language search)
======================================================================
Extracts a CLIP visual embedding for every tracked person, in the same
embedding space as CLIP's text encoder. This is what lets you later type
(or speak) a description like "person in a red jacket" and find matching
tracks -- the text and the images live in the same vector space, so they
can be compared directly with cosine similarity.

This is SEPARATE from extract_embeddings.py (which uses ResNet18 for
cross-camera Re-ID matching) -- different model, different embedding
space, different purpose. Re-ID embeddings answer "is this the same
person as that other sighting?". CLIP embeddings answer "does this
person match this text description?".

Usage:
    python extract_clip_embeddings.py --video cameras/cam1.mp4 \
        --observations outputs/cam1_observations.json \
        --camera_id cam1 --out_dir clip_embeddings

Requires: torch, open_clip_torch, opencv-python, pillow
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import open_clip
from PIL import Image


def build_clip_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)
    return model, preprocess, tokenizer


def extract_clip_embeddings(video_path, observations_path, camera_id,
                             out_dir="clip_embeddings", max_samples_per_track=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _ = build_clip_model(device)

    with open(observations_path) as f:
        observations = json.load(f)

    by_track = defaultdict(list)
    for obs in observations:
        by_track[obs["track_id"]].append(obs)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    track_embeddings = {}

    for track_id, obs_list in by_track.items():
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
            pil_img = Image.fromarray(crop_rgb)
            crops.append(preprocess(pil_img))

        if not crops:
            continue

        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            avg_embedding = features.mean(dim=0)
            avg_embedding = avg_embedding / avg_embedding.norm()

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
    out_file = out_path / f"{camera_id}_clip_embeddings.json"
    with open(out_file, "w") as f:
        json.dump({"camera_id": camera_id, "tracks": track_embeddings}, f)

    print(f"[CatchCam] Extracted CLIP embeddings for {len(track_embeddings)} tracks")
    print(f"[CatchCam] Saved -> {out_file}")
    return track_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--camera_id", required=True)
    parser.add_argument("--out_dir", default="clip_embeddings")
    parser.add_argument("--max_samples_per_track", type=int, default=3)
    args = parser.parse_args()

    extract_clip_embeddings(
        video_path=args.video,
        observations_path=args.observations,
        camera_id=args.camera_id,
        out_dir=args.out_dir,
        max_samples_per_track=args.max_samples_per_track,
    )
