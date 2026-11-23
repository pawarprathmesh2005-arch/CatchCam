"""
CatchCam — Clothing attribute routes (auto-loaded by api.py as `routes_attributes`).
  GET /attributes                     list/filter people by upper/lower colour
  GET /attributes/search              "person in a red jacket and blue jeans" -> ranked tracks
  GET /attributes/verify_matches      each cross-camera Re-ID match + whether the clothing agrees
Needs outputs/<camera>_attributes.json (python -m person_attrs.extract ...).
"""
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from person_attrs.core import match_agreement, normalize_color, parse_description, search_people

router = APIRouter(prefix="/attributes", tags=["attributes"])
DATA_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", "."))
OUTPUTS_DIR = DATA_DIR / "outputs"
MATCHES_FILE = DATA_DIR / "matches" / "cross_camera_matches.json"


def load_attribute_rows():
    rows = []
    for f in sorted(OUTPUTS_DIR.glob("*_attributes.json")) if OUTPUTS_DIR.exists() else []:
        rows += json.loads(f.read_text(encoding="utf-8-sig"))
    return rows


@router.get("")
def list_people(upper_color: Optional[str] = Query(None), lower_color: Optional[str] = Query(None),
                camera_id: Optional[str] = Query(None)):
    rows = load_attribute_rows()
    if upper_color: rows = [r for r in rows if r["upper_color"] == (normalize_color(upper_color) or upper_color.lower())]
    if lower_color: rows = [r for r in rows if r["lower_color"] == (normalize_color(lower_color) or lower_color.lower())]
    if camera_id: rows = [r for r in rows if r["camera_id"] == camera_id]
    return {"count": len(rows), "people": rows}


@router.get("/search")
def search(description: str = Query(..., description="e.g. 'person in a red jacket and blue jeans'"),
           camera_id: Optional[str] = Query(None), limit: int = Query(10, ge=1, le=50)):
    want = parse_description(description)
    if not any(want.values()):
        raise HTTPException(422, "No clothing colour found in the description. Try e.g. 'red jacket' or 'blue jeans'.")
    return {"description": description, "understood_as": want,
            "results": search_people(load_attribute_rows(), description, camera_id, limit)}


@router.get("/verify_matches")
def verify_matches(min_conf: float = Query(0.5, ge=0, le=1)):
    """Marks Re-ID matches whose clothing clearly differs ('conflict') so you can review them first."""
    if not MATCHES_FILE.exists():
        raise HTTPException(404, f"Matches file not found: {MATCHES_FILE}")
    people = {(r["camera_id"], r["track_id"]): r for r in load_attribute_rows()}
    matches = json.loads(MATCHES_FILE.read_text(encoding="utf-8-sig"))
    out = []
    for m in matches:
        verdict = match_agreement(people.get((m["camera_a"], m["track_a"])), people.get((m["camera_b"], m["track_b"])), min_conf)
        out.append({**m, "clothing": verdict})
    counts = {k: sum(1 for o in out if o["clothing"]["verdict"] == k) for k in ("consistent", "conflict", "unknown")}
    return {"counts": counts, "matches": sorted(out, key=lambda o: o["clothing"]["verdict"] != "conflict")}
