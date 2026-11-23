"""
CatchCam — Week 2: Cross-Camera Re-ID Matching
=================================================
Compares every tracked person's appearance embedding across 2+ cameras and
reports the most likely matches -- i.e. "this person in cam1 is probably
this person in cam2".

This is the core novelty step: turning independent per-camera tracks into
cross-camera identity links, which Week 3's Space-Time Graph will chain
together into full journeys.

Usage:
    python match_cameras.py --embeddings embeddings/cam1_embeddings.json embeddings/cam2_embeddings.json embeddings/cam3_embeddings.json --threshold 0.75

Requires: numpy (already in your venv)
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def load_embeddings(paths):
    cameras = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        cam_id = data["camera_id"]
        tracks = data["tracks"]
        cameras[cam_id] = {
            int(tid): np.array(info["embedding"])
            for tid, info in tracks.items()
        }
        # keep the metadata too (timestamps) for reporting
        cameras[cam_id + "__meta"] = tracks
    return cameras


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def match_cameras(embedding_paths, threshold: float = 0.75, out_dir: str = "matches"):
    cameras = load_embeddings(embedding_paths)
    cam_ids = [k for k in cameras.keys() if not k.endswith("__meta")]

    print(f"[CatchCam] Loaded {len(cam_ids)} cameras: {cam_ids}")
    for cam_id in cam_ids:
        print(f"  {cam_id}: {len(cameras[cam_id])} tracked people")

    all_matches = []

    # Compare every unique pair of cameras
    for cam_a, cam_b in combinations(cam_ids, 2):
        tracks_a = cameras[cam_a]
        tracks_b = cameras[cam_b]
        meta_a = cameras[cam_a + "__meta"]
        meta_b = cameras[cam_b + "__meta"]

        for tid_a, emb_a in tracks_a.items():
            best_match = None
            best_score = -1.0

            for tid_b, emb_b in tracks_b.items():
                score = cosine_similarity(emb_a, emb_b)
                if score > best_score:
                    best_score = score
                    best_match = tid_b

            if best_match is not None and best_score >= threshold:
                all_matches.append({
                    "camera_a": cam_a,
                    "track_a": tid_a,
                    "camera_b": cam_b,
                    "track_b": best_match,
                    "similarity": round(best_score, 4),
                    "a_last_seen": meta_a[str(tid_a)]["last_seen_timestamp"],
                    "b_first_seen": meta_b[str(best_match)]["first_seen_timestamp"],
                })

    # Sort strongest matches first
    all_matches.sort(key=lambda m: -m["similarity"])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "cross_camera_matches.json"
    with open(out_file, "w") as f:
        json.dump(all_matches, f, indent=2)

    print(f"\n[CatchCam] Found {len(all_matches)} candidate matches above threshold {threshold}")
    print(f"[CatchCam] Saved -> {out_file}\n")

    print("Top matches:")
    for m in all_matches[:10]:
        print(f"  {m['camera_a']} track {m['track_a']}  <->  "
              f"{m['camera_b']} track {m['track_b']}  "
              f"(similarity: {m['similarity']})")

    return all_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", nargs="+", required=True,
                         help="Paths to 2+ camera embedding JSON files")
    parser.add_argument("--threshold", type=float, default=0.75,
                         help="Minimum cosine similarity to count as a match (0-1)")
    parser.add_argument("--out_dir", default="matches")
    args = parser.parse_args()

    match_cameras(
        embedding_paths=args.embeddings,
        threshold=args.threshold,
        out_dir=args.out_dir,
    )
