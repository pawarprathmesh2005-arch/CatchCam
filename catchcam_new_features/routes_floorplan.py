"""
CatchCam — Floor-plan routes (auto-loaded by api.py as `routes_floorplan`). Needs <data>/floorplan.json (see floorplan/core.py).
  GET /floorplan/config                 cameras on the plan + calibration quality
  GET /floorplan/journey                a person's full journey as points on the plan (+ unexplained gaps)
  GET /floorplan/render                 the same as a PNG picture
"""
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from build_graph import build_graph, node_id
from floorplan.core import calibration_report, journey_on_plan, load_config, render_png
from missing_camera import find_coverage_gaps, get_full_journey

router = APIRouter(prefix="/floorplan", tags=["floorplan"])
DATA_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", "."))
OUTPUTS_DIR = DATA_DIR / "outputs"
MATCHES_FILE = DATA_DIR / "matches" / "cross_camera_matches.json"
CONFIG_FILE = DATA_DIR / "floorplan.json"


def _cfg():
    if not CONFIG_FILE.exists():
        raise HTTPException(404, f"floorplan.json not found in {DATA_DIR.resolve()} -- see floorplan/core.py for the format")
    try:
        return load_config(CONFIG_FILE)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"floorplan.json is invalid: {e}")


def _plan(camera_id, track_id, max_normal_gap, critical_gap=45.0):
    cfg = _cfg()
    files = sorted(OUTPUTS_DIR.glob("*_observations.json")) if OUTPUTS_DIR.exists() else []
    if not files:
        raise HTTPException(404, "No observation data found")
    if not MATCHES_FILE.exists():
        raise HTTPException(404, f"Matches file not found: {MATCHES_FILE}")
    G = build_graph([str(f) for f in files], str(MATCHES_FILE))
    if node_id(camera_id, track_id) not in G.nodes:
        raise HTTPException(404, f"No sighting found for {camera_id} track {track_id}")
    journey = get_full_journey(G, camera_id, track_id)
    gaps = find_coverage_gaps(journey, max_normal_gap)
    for g in gaps:
        g["severity"] = "critical" if g["gap_seconds"] > critical_gap else "suspicious"
    obs = []
    for f in files:
        obs += json.loads(f.read_text(encoding="utf-8-sig"))
    return cfg, journey, gaps, journey_on_plan(journey, gaps, cfg, obs)


@router.get("/config")
def floorplan_config():
    cfg = _cfg()
    return {"width": cfg["width"], "height": cfg["height"], "has_background_image": bool(cfg.get("image")),
            "cameras": calibration_report(cfg)}


@router.get("/journey")
def floorplan_journey(camera_id: str = Query(...), track_id: int = Query(...),
                      max_normal_gap: float = Query(15.0, ge=0), critical_gap: float = Query(45.0, ge=0)):
    cfg, journey, gaps, plan = _plan(camera_id, track_id, max_normal_gap, critical_gap)
    return {"camera_id": camera_id, "track_id": track_id, "width": cfg["width"], "height": cfg["height"],
            "gaps": gaps, **plan}


@router.get("/render")
def floorplan_render(camera_id: str = Query(...), track_id: int = Query(...), max_normal_gap: float = Query(15.0, ge=0),
                     critical_gap: float = Query(45.0, ge=0)):
    cfg, journey, gaps, plan = _plan(camera_id, track_id, max_normal_gap, critical_gap)
    png = render_png(cfg, plan, DATA_DIR, title=f"{camera_id} track {track_id}: " + " > ".join(h["camera_id"] for h in journey))
    return Response(content=png, media_type="image/png")
