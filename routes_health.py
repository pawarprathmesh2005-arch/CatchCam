"""
CatchCam — Missing-Camera Detection API routes
==================================================
Auto-loaded by api.py if this file is present. Exposes:
  GET /coverage_gaps   -- full journey of one person + any suspiciously long unexplained gaps in it
  GET /camera_health   -- which live cameras are currently reporting (and which have gone DOWN)

No new dependencies beyond what api.py already uses.

(Extends the earlier version: the original response keys are unchanged; the new keys --
 journey, severity, has_blind_spot, is_down, num_down -- are additions the dashboard uses.)
"""
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from build_graph import build_graph
from missing_camera import get_full_journey, find_coverage_gaps, check_live_camera_health

router = APIRouter()

DATA_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", "."))
OUTPUTS_DIR = DATA_DIR / "outputs"
MATCHES_FILE = DATA_DIR / "matches" / "cross_camera_matches.json"


def _discover_observation_files():
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(OUTPUTS_DIR.glob("*_observations.json"))


@router.get("/coverage_gaps")
def coverage_gaps(
    camera_id: str = Query(..., description="Camera of any sighting in the journey"),
    track_id: int = Query(..., description="Track ID of that sighting"),
    max_normal_gap: float = Query(15.0, description="Seconds -- gaps longer than this are flagged"),
    critical_gap: float = Query(45.0, description="Gaps longer than this are marked 'critical' instead of 'suspicious'"),
):
    observation_files = [str(f) for f in _discover_observation_files()]
    if not observation_files:
        raise HTTPException(404, "No observation data found")
    if not MATCHES_FILE.exists():
        raise HTTPException(404, f"Matches file not found: {MATCHES_FILE}")

    G = build_graph(observation_files, str(MATCHES_FILE))
    journey = get_full_journey(G, camera_id, track_id)   # traced back to first appearance, then forward
    if not journey:
        raise HTTPException(404, f"No sighting found for {camera_id} track {track_id}")

    gaps = find_coverage_gaps(journey, max_normal_gap)
    for g in gaps:
        g["severity"] = "critical" if g["gap_seconds"] > critical_gap else "suspicious"
    return {"journey_length": len(journey), "max_normal_gap": max_normal_gap,
            "gaps": gaps, "journey": journey, "has_blind_spot": bool(gaps)}


@router.get("/camera_health")
def camera_health(stale_after: float = Query(10.0, description="Seconds of silence before a camera is flagged DOWN")):
    result = check_live_camera_health(str(OUTPUTS_DIR), stale_after)
    for cam in result["cameras"]:
        cam["is_down"] = cam["status"].startswith("DOWN")
    result["num_down"] = sum(c["is_down"] for c in result["cameras"])
    result["stale_after"] = stale_after
    return result
