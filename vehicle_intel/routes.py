"""
FastAPI endpoints for vehicle intelligence. Loaded automatically by api.py through routes_vehicles.py.
"""
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from .core import plate_distance

router = APIRouter(prefix="/vehicles", tags=["vehicles"])
OUTPUTS_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", ".")) / "outputs"


def load_vehicles():
    rows = []
    for f in sorted(OUTPUTS_DIR.glob("*_vehicles.json")) if OUTPUTS_DIR.exists() else []:
        rows += json.loads(f.read_text())
    return rows


def _norm(p: str) -> str:
    return "".join(ch for ch in (p or "").upper() if ch.isalnum())


def search_vehicles(vehicle_type=None, color=None, plate=None, camera_id=None, max_errors=1):
    """Plain function (no FastAPI parameter objects) so both endpoints can share it safely."""
    rows = load_vehicles()
    if vehicle_type: rows = [r for r in rows if r["vehicle_type"] == vehicle_type.lower()]
    if color: rows = [r for r in rows if r["color"] == color.lower()]
    if camera_id: rows = [r for r in rows if r["camera_id"] == camera_id]
    if plate:
        q = _norm(plate)
        rows = [r for r in rows if r["plate"] and (q in _norm(r["plate"]) or
                (len(q) >= 8 and plate_distance(q, _norm(r["plate"])) <= max_errors))]
    return rows


@router.get("")
def list_vehicles(vehicle_type: Optional[str] = Query(None, description="car | motorcycle | bus | truck"),
                  color: Optional[str] = Query(None), plate: Optional[str] = Query(None, description="full or partial plate"),
                  camera_id: Optional[str] = Query(None),
                  max_errors: int = Query(1, ge=0, le=3, description="OCR mistakes tolerated when the plate is complete")):
    rows = search_vehicles(vehicle_type, color, plate, camera_id, max_errors)
    return {"count": len(rows), "vehicles": rows}


@router.get("/journey")
def vehicle_journey(plate: str = Query(..., description="Plate to follow across cameras"),
                    max_errors: int = Query(1, ge=0, le=3),
                    max_normal_gap: float = Query(15.0)):
    """Every sighting of this plate across cameras, in time order, with missing-camera gap analysis."""
    hits = search_vehicles(plate=plate, max_errors=max_errors)
    if not hits:
        raise HTTPException(404, f"No sightings of plate {plate}")
    hits = sorted(hits, key=lambda r: (r["first_seen_timestamp"] if r["first_seen_timestamp"] is not None else 0))
    out = {"plate": plate, "sightings": hits}
    try:                                    # same gap check the people journeys use (missing_camera.py)
        from missing_camera import find_coverage_gaps
        out["gaps"] = find_coverage_gaps(hits, max_normal_gap)
    except ImportError:
        out["gaps"] = []
    return out
