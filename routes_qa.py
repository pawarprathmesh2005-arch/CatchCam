"""
CatchCam — Conversational Q&A route (auto-loaded by api.py as `routes_qa`).
  GET /qa?question=Where did the person in the red jacket go?
Returns {"answer": ..., "speak": ..., "intent": ..., "subject": ..., "data": {...}}; the dashboard shows `answer`
and reads `speak` aloud with the browser's speech synthesis. Answers are built only from the project's own data.
"""
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

from qa_engine.engine import EXAMPLES, Store, ask

router = APIRouter(tags=["qa"])
DATA_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", "."))


@router.get("/qa")
def question_answer(question: str = Query(..., description="A question about the footage, typed or transcribed from speech"),
                    max_normal_gap: float = Query(15.0, ge=0),
                    context_camera: Optional[str] = Query(None, description="from the previous answer's context.camera_id"),
                    context_track: Optional[int] = Query(None, description="from the previous answer's context.track_id"),
                    context_plate: Optional[str] = Query(None, description="from the previous answer's context.plate")):
    """Follow-ups: send the previous answer's `context` back as context_* so 'he' / 'that car' / 'them' resolve."""
    ctx = None
    if context_plate:
        ctx = {"plate": context_plate}
    elif context_camera is not None and context_track is not None:
        ctx = {"camera_id": context_camera, "track_id": context_track}
    return ask(question, Store(DATA_DIR), max_normal_gap, ctx)


@router.get("/qa/examples")
def qa_examples():
    return {"examples": EXAMPLES}
