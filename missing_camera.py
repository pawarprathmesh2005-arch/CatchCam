"""
CatchCam — Missing-Camera Detection
=======================================
Two independent checks:

1. gap analysis (offline): looks at a reconstructed journey and flags any
   hop-to-hop time gap that's suspiciously long -- e.g. someone vanishes
   for 45 seconds between two cameras that are normally only 10 seconds'
   walk apart. This usually means either a blind spot in your camera
   coverage, or a missed/failed match that a human should double-check.

2. live health check (online): for realtime_track.py's live output files,
   flags any camera whose observations file hasn't been updated recently
   -- meaning that camera has likely gone down or disconnected.

Usage:
    # Gap analysis on a specific journey
    python missing_camera.py gaps --observations outputs/cam1_observations.json outputs/cam2_observations.json \
        --matches matches/cross_camera_matches.json \
        --start_camera cam1 --start_track 5 --max_normal_gap 15

    # Live health check across your outputs folder
    python missing_camera.py health --outputs_dir outputs --stale_after 10
"""
import argparse
import json
import time
from pathlib import Path

from build_graph import build_graph, reconstruct_journey, reconstruct_journey_backward


def get_full_journey(G, start_camera, start_track):
    backward = reconstruct_journey_backward(G, start_camera, start_track)
    if not backward:
        return reconstruct_journey(G, start_camera, start_track)
    true_start = backward[0]
    return reconstruct_journey(G, true_start["camera_id"], true_start["track_id"])


def find_coverage_gaps(journey, max_normal_gap: float = 15.0):
    """
    Walks through a journey's hops and flags any gap between one camera's
    last_seen and the next camera's first_seen that exceeds
    max_normal_gap seconds. Tune max_normal_gap to roughly match the real
    walking time between your most distant adjacent cameras.
    """
    gaps = []
    for i in range(len(journey) - 1):
        current_hop = journey[i]
        next_hop = journey[i + 1]
        gap_duration = next_hop["first_seen_timestamp"] - current_hop["last_seen_timestamp"]

        if gap_duration > max_normal_gap:
            gaps.append({
                "from_camera": current_hop["camera_id"],
                "from_track": current_hop["track_id"],
                "last_seen": current_hop["last_seen_timestamp"],
                "to_camera": next_hop["camera_id"],
                "to_track": next_hop["track_id"],
                "next_seen": next_hop["first_seen_timestamp"],
                "gap_seconds": round(gap_duration, 2),
                "flag": (
                    f"Unaccounted for {gap_duration:.1f}s between {current_hop['camera_id']} "
                    f"and {next_hop['camera_id']} -- possible blind spot in coverage, "
                    f"or a missed match that's worth checking manually."
                ),
            })
    return gaps


def check_live_camera_health(outputs_dir: str, stale_after: float = 10.0):
    """
    Checks every *_observations_live.json file (written by realtime_track.py)
    and flags any camera whose file hasn't been modified in the last
    `stale_after` seconds -- meaning that camera's process has likely
    stopped, crashed, or the physical camera disconnected.
    """
    out_path = Path(outputs_dir)
    live_files = sorted(out_path.glob("*_observations_live.json"))

    if not live_files:
        return {"status": "no_live_cameras", "cameras": []}

    now = time.time()
    results = []
    for f in live_files:
        camera_id = f.name.replace("_observations_live.json", "")
        age_seconds = now - f.stat().st_mtime
        is_stale = age_seconds > stale_after
        results.append({
            "camera_id": camera_id,
            "last_update_seconds_ago": round(age_seconds, 1),
            "status": "DOWN -- no recent updates" if is_stale else "OK -- live",
        })

    return {"status": "checked", "cameras": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    gaps_parser = subparsers.add_parser("gaps", help="Find coverage gaps in a reconstructed journey")
    gaps_parser.add_argument("--observations", nargs="+", required=True)
    gaps_parser.add_argument("--matches", required=True)
    gaps_parser.add_argument("--start_camera", required=True)
    gaps_parser.add_argument("--start_track", type=int, required=True)
    gaps_parser.add_argument("--max_normal_gap", type=float, default=15.0)

    health_parser = subparsers.add_parser("health", help="Check live camera feed health")
    health_parser.add_argument("--outputs_dir", default="outputs")
    health_parser.add_argument("--stale_after", type=float, default=10.0)

    args = parser.parse_args()

    if args.command == "gaps":
        G = build_graph(args.observations, args.matches)
        journey = get_full_journey(G, args.start_camera, args.start_track)
        print(f"[CatchCam] Journey has {len(journey)} hops")
        gaps = find_coverage_gaps(journey, args.max_normal_gap)
        if not gaps:
            print(f"[CatchCam] No coverage gaps found (threshold: {args.max_normal_gap}s)")
        else:
            print(f"[CatchCam] Found {len(gaps)} coverage gap(s):")
            for g in gaps:
                print(f"  {g['flag']}")

    elif args.command == "health":
        result = check_live_camera_health(args.outputs_dir, args.stale_after)
        if result["status"] == "no_live_cameras":
            print("[CatchCam] No live camera files found. Is realtime_track.py running?")
        else:
            for cam in result["cameras"]:
                print(f"  {cam['camera_id']}: {cam['status']} "
                      f"(last update {cam['last_update_seconds_ago']}s ago)")
