"""
CatchCam — watchlist alert routes (loaded by routes_web; no change to api.py needed).

  GET    /alerts                     the alerts (newest first), the watchlist, and which notification channels are set up
  GET    /alerts/count               just the number of unacknowledged alerts (the header badge polls this)
  POST   /alerts/watch               add to the watchlist:  ?kind=plate|clothing|gap|camera_down&value=...
  DELETE /alerts/watch/{item_id}     remove a watchlist item
  POST   /alerts/ack                 acknowledge one alert (?alert_id=...) or all (?everything=true)

Where alerts are sent (Slack/Discord/Telegram...) is set with environment variables or alerts_config.json, never through this API.
"""
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from alerts import core as al

router = APIRouter(tags=["alerts"])
DATA_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", "."))


@router.get("/alerts")
def alerts():
    return al.overview(DATA_DIR)


@router.get("/alerts/count")
def count():
    return {"unacknowledged": al.overview(DATA_DIR)["unacknowledged"]}


@router.post("/alerts/watch")
def add_watch(kind: str = Query(...), value: str = Query(""), label: Optional[str] = Query(None)):
    try:
        return al.add_item(DATA_DIR, kind, value, label)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.delete("/alerts/watch/{item_id}")
def remove_watch(item_id: str):
    if not al.remove_item(DATA_DIR, item_id):
        raise HTTPException(404, "No such watchlist item")
    return {"removed": item_id}


@router.post("/alerts/ack")
def ack(alert_id: Optional[str] = Query(None), everything: bool = Query(False)):
    if not alert_id and not everything:
        raise HTTPException(422, "Give an alert_id, or everything=true.")
    return {"acknowledged": al.acknowledge(DATA_DIR, alert_id, everything)}
