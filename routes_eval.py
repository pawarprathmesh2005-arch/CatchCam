"""
CatchCam — accuracy evaluation routes (loaded by routes_web, no change to api.py needed).

  GET  /eval/tracks      every track with its thumbnail, the label you gave it, and which journey CatchCam put it on
  POST /eval/labels      save labels: JSON body {"labels": [{"camera_id": "cam1", "track_id": 3, "person": "P1"}, ...]}
  GET  /eval/run         score CatchCam against your labels
  GET  /eval/report.md   the same result as a Markdown write-up you can paste into the project report
"""
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from evaluation import core as ev

router = APIRouter(tags=["evaluation"])
DATA_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", "."))


@router.get("/eval/tracks")
def tracks():
    return ev.track_list(DATA_DIR)


@router.post("/eval/labels")
async def save_labels(request: Request):
    raw = await request.body()
    if len(raw) > 2_000_000:
        raise HTTPException(413, "Too many labels in one request.")
    try:
        body = json.loads(raw or b"{}")
        rows = body["labels"]
        if not isinstance(rows, list):
            raise TypeError
    except (ValueError, KeyError, TypeError):
        raise HTTPException(422, 'Send JSON like {"labels": [{"camera_id": "cam1", "track_id": 3, "person": "P1"}]}.')
    try:
        kept = ev.save_labels(DATA_DIR, rows)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"saved": kept}


@router.get("/eval/run")
def run():
    return ev.evaluate(DATA_DIR)


@router.get("/eval/report.md")
def report():
    text = ev.to_markdown(ev.evaluate(DATA_DIR))
    return Response(content=text.encode("utf-8"), media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="catchcam_evaluation.md"'})
