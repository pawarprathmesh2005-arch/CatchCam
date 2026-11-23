"""
CatchCam — Journey Video Reconstruction
===========================================
Given a starting (camera, track_id) sighting, this:
  1. Traces the FULL journey (backward to find where they first entered,
     forward to see everywhere they went after)
  2. Trims each camera's source video down to just the segment where
     that person was actually visible (with a small padding buffer)
  3. Normalizes every segment to the same resolution (cameras have
     different native sizes) and overlays a camera/timestamp label
  4. Stitches everything into one continuous MP4, in chronological order

This is what turns a text-based journey ("cam1 -> cam3 -> cam5") into an
actual watchable video of the person's path through the building.

Usage:
    python journey_video.py --observations outputs/cam1_observations.json outputs/cam2_observations.json \
        --matches matches/cross_camera_matches.json \
        --camera_videos camera_videos.json \
        --start_camera cam1 --start_track 5 \
        --out journey_output.mp4

camera_videos.json maps each camera_id to its source video file, e.g.:
    {"cam1": "cameras/cam1_topleft.mp4", "cam2": "cameras/cam2_topright.mp4"}

Requires: ffmpeg installed and on PATH, networkx (pip install networkx)
Also needs build_graph.py in the same folder.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from build_graph import build_graph, reconstruct_journey, reconstruct_journey_backward, node_id


def get_full_journey(G, start_camera, start_track):
    """
    Traces backward first to find the true starting point, then walks
    forward from there -- which already captures the complete chain in
    one pass (no need to merge two separate lists).
    """
    backward = reconstruct_journey_backward(G, start_camera, start_track)
    if not backward:
        return reconstruct_journey(G, start_camera, start_track)

    true_start = backward[0]
    return reconstruct_journey(G, true_start["camera_id"], true_start["track_id"])


def extract_and_normalize_clip(video_path, start_time, end_time, out_path,
                                target_width=640, target_height=360,
                                label=None, pad_seconds=0.5):
    """
    Cuts [start_time, end_time] (plus a small padding buffer) out of
    video_path, scales/pads it to a fixed resolution so all cameras'
    clips can later be joined cleanly, and burns in a text label
    (typically "CAM1 | 12.3s") so it's clear which segment is playing.
    """
    padded_start = max(0.0, start_time - pad_seconds)
    duration = (end_time - start_time) + 2 * pad_seconds

    vf_filters = [
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease",
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black",
    ]
    if label:
        safe_label = label.replace(":", "\\:").replace("'", "")
        vf_filters.append(
            f"drawtext=text='{safe_label}':fontcolor=white:fontsize=18:"
            f"box=1:boxcolor=black@0.6:boxborderw=5:x=10:y=10"
        )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(padded_start),
        "-i", str(video_path),
        "-t", str(max(duration, 0.5)),
        "-vf", ",".join(vf_filters),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",  # drop audio -- CCTV clips rarely have useful audio, and it
                # simplifies concatenation (mismatched audio streams are a
                # common cause of concat failures)
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video_path}:\n{result.stderr[-800:]}")


def build_journey_video(observation_paths, matches_path, camera_videos_path,
                         start_camera, start_track, out_path,
                         target_width=640, target_height=360, pad_seconds=0.5):
    # encoding="utf-8-sig" transparently handles a UTF-8 BOM if present
    # (common when a JSON file is created via PowerShell's Out-File),
    # while still reading perfectly normal UTF-8 files with no BOM.
    with open(camera_videos_path, encoding="utf-8-sig") as f:
        camera_videos = json.load(f)

    G = build_graph(observation_paths, matches_path)
    journey = get_full_journey(G, start_camera, start_track)

    if not journey:
        raise ValueError(f"No sighting found for {start_camera} track {start_track}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        segment_files = []

        for i, hop in enumerate(journey):
            cam_id = hop["camera_id"]
            if cam_id not in camera_videos:
                print(f"[CatchCam] WARNING: no video path for {cam_id} in camera_videos.json -- skipping this hop")
                continue

            video_path = camera_videos[cam_id]
            segment_out = tmp_path / f"segment_{i:03d}.mp4"
            label = f"{cam_id.upper()} | {hop['first_seen_timestamp']:.1f}s-{hop['last_seen_timestamp']:.1f}s"

            print(f"[CatchCam] Extracting segment {i+1}/{len(journey)}: {label}")
            extract_and_normalize_clip(
                video_path, hop["first_seen_timestamp"], hop["last_seen_timestamp"],
                segment_out, target_width, target_height, label, pad_seconds,
            )
            segment_files.append(segment_out)

        if not segment_files:
            raise RuntimeError("No segments could be extracted -- check camera_videos.json paths")

        concat_list = tmp_path / "concat_list.txt"
        with open(concat_list, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        print(f"[CatchCam] Stitching {len(segment_files)} segments into final video...")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr[-800:]}")

    print(f"[CatchCam] Journey video saved -> {out_path}")
    return journey


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", nargs="+", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--camera_videos", required=True,
                         help="Path to camera_videos.json mapping camera_id -> video file")
    parser.add_argument("--start_camera", required=True)
    parser.add_argument("--start_track", type=int, required=True)
    parser.add_argument("--out", default="journey_output.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--pad_seconds", type=float, default=0.5)
    args = parser.parse_args()

    build_journey_video(
        observation_paths=args.observations,
        matches_path=args.matches,
        camera_videos_path=args.camera_videos,
        start_camera=args.start_camera,
        start_track=args.start_track,
        out_path=args.out,
        target_width=args.width,
        target_height=args.height,
        pad_seconds=args.pad_seconds,
    )
