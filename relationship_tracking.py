"""
CatchCam — Multi-Object Relationship Tracking
=================================================
Detects when two tracked people are close together in the same camera for
a sustained period -- a proxy for "these two people are together" (walking
together, meeting, talking), as opposed to briefly passing by each other.

This feeds two later features:
  - A "relationship" layer on the dashboard (show who was seen together)
  - The Incident Reconstruction module, which uses these events to narrate
    things like "14:10 -- Person meets another individual"

Method: for every frame, compute the distance between every pair of people
visible in that camera, normalized by their bounding box size (so it's fair
whether someone's near or far from the camera). Pairs that stay close for
at least --min_duration seconds are reported as a relationship event.

Usage:
    python relationship_tracking.py --observations outputs/cam1_observations.json --camera_id cam1 --out_dir relationships

    # Across multiple cameras at once:
    python relationship_tracking.py --observations outputs/cam1_observations.json outputs/cam2_observations.json --out_dir relationships

Requires: nothing beyond the standard library (no models, no downloads)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def bbox_center_and_height(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    height = max(y2 - y1, 1)  # avoid divide-by-zero
    return center, height


def normalized_distance(bbox_a, bbox_b):
    """
    Distance between two boxes' centers, normalized by their average height.
    A normalized distance of ~1.0 roughly means "about one person-height
    apart" -- a useful, scale-invariant proxy for real-world closeness.
    """
    center_a, h_a = bbox_center_and_height(bbox_a)
    center_b, h_b = bbox_center_and_height(bbox_b)
    pixel_dist = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
    avg_height = (h_a + h_b) / 2
    return pixel_dist / avg_height


def find_relationships(observation_paths, proximity_threshold: float = 2.0,
                        min_duration: float = 1.0, out_dir: str = "relationships"):
    """
    proximity_threshold: max normalized distance to count as "close"
                          (2.0 = roughly within 2 person-heights)
    min_duration: minimum seconds a pair must stay close to count as a
                  real relationship, not just a brief pass-by
    """
    all_events = []

    for obs_path in observation_paths:
        with open(obs_path) as f:
            observations = json.load(f)

        if not observations:
            continue
        camera_id = observations[0]["camera_id"]

        # Group by frame -> list of (track_id, bbox, timestamp)
        by_frame = defaultdict(list)
        for obs in observations:
            by_frame[obs["frame"]].append(obs)

        # For each pair of tracks, record every frame they were "close"
        pair_close_frames = defaultdict(list)  # (track_a, track_b) -> [(frame, timestamp, distance)]

        for frame, obs_list in sorted(by_frame.items()):
            for i in range(len(obs_list)):
                for j in range(i + 1, len(obs_list)):
                    a, b = obs_list[i], obs_list[j]
                    if a["track_id"] == b["track_id"]:
                        continue
                    dist = normalized_distance(a["bbox_xyxy"], b["bbox_xyxy"])
                    if dist <= proximity_threshold:
                        pair_key = tuple(sorted([a["track_id"], b["track_id"]]))
                        pair_close_frames[pair_key].append((frame, a["timestamp"], dist))

        # Merge consecutive close-frames into events, keep only sustained ones
        for (track_a, track_b), records in pair_close_frames.items():
            records.sort(key=lambda r: r[0])
            event_start = records[0]
            prev = records[0]
            distances = [records[0][2]]

            def flush_event(start, end, distances):
                duration = end[1] - start[1]
                if duration >= min_duration:
                    all_events.append({
                        "camera_id": camera_id,
                        "track_a": track_a,
                        "track_b": track_b,
                        "start_frame": start[0],
                        "end_frame": end[0],
                        "start_timestamp": round(start[1], 3),
                        "end_timestamp": round(end[1], 3),
                        "duration_seconds": round(duration, 3),
                        "avg_proximity": round(sum(distances) / len(distances), 3),
                        "min_proximity": round(min(distances), 3),
                    })

            for record in records[1:]:
                frame_gap = record[0] - prev[0]
                if frame_gap > 5:  # gap of >5 frames = treat as a new, separate event
                    flush_event(event_start, prev, distances)
                    event_start = record
                    distances = []
                distances.append(record[2])
                prev = record
            flush_event(event_start, prev, distances)

    all_events.sort(key=lambda e: e["start_timestamp"])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "relationships.json"
    with open(out_file, "w") as f:
        json.dump(all_events, f, indent=2)

    print(f"[CatchCam] Found {len(all_events)} sustained proximity events "
          f"(threshold={proximity_threshold}, min_duration={min_duration}s)")
    print(f"[CatchCam] Saved -> {out_file}\n")

    for e in all_events[:10]:
        print(f"  {e['camera_id']}: track {e['track_a']} <-> track {e['track_b']}  "
              f"({e['start_timestamp']:.2f}s-{e['end_timestamp']:.2f}s, "
              f"{e['duration_seconds']:.1f}s together, "
              f"avg proximity {e['avg_proximity']})")

    return all_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", nargs="+", required=True,
                         help="One or more camera observation JSON files")
    parser.add_argument("--proximity_threshold", type=float, default=2.0,
                         help="Max normalized distance to count as 'close' "
                              "(in units of average person-height)")
    parser.add_argument("--min_duration", type=float, default=1.0,
                         help="Minimum seconds together to count as a real event")
    parser.add_argument("--out_dir", default="relationships")
    args = parser.parse_args()

    find_relationships(
        observation_paths=args.observations,
        proximity_threshold=args.proximity_threshold,
        min_duration=args.min_duration,
        out_dir=args.out_dir,
    )
