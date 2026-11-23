"""
CatchCam — incident report route (loaded by routes_web; no change to api.py needed).
  GET /report/journey.pdf?camera_id=cam1&track_id=3     a PDF for one person's journey. Needs: pip install reportlab
"""
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

router = APIRouter(tags=["report"])
DATA_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", "."))


@router.get("/report/journey.pdf")
def journey_report(camera_id: str = Query(...), track_id: int = Query(...), max_normal_gap: float = Query(15.0, ge=0),
                   critical_gap: float = Query(45.0, ge=0)):
    try:
        from reports.core import build_report
    except ModuleNotFoundError as e:
        raise HTTPException(501, f"The report needs the '{e.name}' package. Run:  pip install reportlab")
    try:
        pdf = build_report(DATA_DIR, camera_id, track_id, max_normal_gap, critical_gap)
    except LookupError as e:
        raise HTTPException(404, str(e))
    safe = "".join(ch for ch in f"{camera_id}_track{track_id}" if ch.isalnum() or ch in "_-")
    return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="catchcam_report_{safe}.pdf"'})
