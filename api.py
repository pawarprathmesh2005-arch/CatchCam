"""
CatchCam — Backend API
==========================
Ties every module together into one FastAPI server. Core endpoints live
in this file; extra features live in their own files (routes_*.py) and
are picked up AUTOMATICALLY if present -- drop a new routes_*.py file in
this folder and restart, no edits to api.py needed.

Usage:
    uvicorn api:app --reload --port 8000
    then open http://localhost:8000/docs
"""
import importlib
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from build_graph import build_graph, reconstruct_journey, reconstruct_journey_backward, node_id
from incident_reconstruction import build_incident_timeline, narrate_timeline_template

app = FastAPI(title="CatchCam API", version="0.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", "."))
OUTPUTS_DIR = DATA_DIR / "outputs"
MATCHES_FILE = DATA_DIR / "matches" / "cross_camera_matches.json"
RELATIONSHIPS_FILE = DATA_DIR / "relationships" / "relationships.json"
CLIP_EMBEDDINGS_DIR = DATA_DIR / "clip_embeddings"


def discover_observation_files():
    """Finds every *_observations.json file in the outputs folder."""
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(OUTPUTS_DIR.glob("*_observations.json"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_dir": str(DATA_DIR.resolve()),
        "observation_files_found": len(discover_observation_files()),
    }


@app.get("/cameras")
def list_cameras():
    cameras = []
    for f in discover_observation_files():
        with open(f, encoding="utf-8-sig") as fh:
            obs = json.load(fh)
        if obs:
            track_ids = sorted(set(o["track_id"] for o in obs))
            cameras.append({
                "camera_id": obs[0]["camera_id"],
                "observation_file": str(f),
                "num_tracks": len(track_ids),
                "track_ids": track_ids,
            })
    return {"cameras": cameras}


@app.get("/journey")
def get_journey(
    camera_id: str = Query(...),
    track_id: int = Query(...),
    direction: str = Query("forward", pattern="^(forward|backward)$"),
):
    observation_files = [str(f) for f in discover_observation_files()]
    if not observation_files:
        raise HTTPException(404, "No observation data found")
    if not MATCHES_FILE.exists():
        raise HTTPException(404, f"Matches file not found: {MATCHES_FILE}")

    G = build_graph(observation_files, str(MATCHES_FILE))
    if node_id(camera_id, track_id) not in G.nodes:
        raise HTTPException(404, f"No such sighting: camera_id={camera_id}, track_id={track_id}")

    if direction == "forward":
        journey = reconstruct_journey(G, camera_id, track_id)
    else:
        journey = reconstruct_journey_backward(G, camera_id, track_id)
    return {"camera_id": camera_id, "track_id": track_id, "direction": direction, "journey": journey}


@app.post("/journey_video")
def create_journey_video(camera_id: str = Query(...), track_id: int = Query(...)):
    camera_videos_file = DATA_DIR / "camera_videos.json"
    if not camera_videos_file.exists():
        raise HTTPException(404, "camera_videos.json not found -- create it mapping "
                                  "each camera_id to its source video file")
    observation_files = [str(f) for f in discover_observation_files()]
    if not observation_files:
        raise HTTPException(404, "No observation data found")
    if not MATCHES_FILE.exists():
        raise HTTPException(404, f"Matches file not found: {MATCHES_FILE}")

    from journey_video import build_journey_video

    out_dir = DATA_DIR / "journey_videos"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"journey_{camera_id}_{track_id}.mp4"
    try:
        journey = build_journey_video(
            observation_paths=observation_files, matches_path=str(MATCHES_FILE),
            camera_videos_path=str(camera_videos_file),
            start_camera=camera_id, start_track=track_id, out_path=str(out_path),
        )
    except Exception as e:
        raise HTTPException(500, f"Video reconstruction failed: {e}")
    return {"journey": journey, "video_path": str(out_path), "video_url": f"/journey_video_file/{out_path.name}"}


@app.get("/journey_video_file/{filename}")
def get_journey_video_file(filename: str):
    video_path = DATA_DIR / "journey_videos" / filename
    if not video_path.exists():
        raise HTTPException(404, "Video not found")
    with open(video_path, "rb") as f:
        return Response(content=f.read(), media_type="video/mp4")


@app.get("/relationships")
def get_relationships(min_duration: float = Query(0.0)):
    if not RELATIONSHIPS_FILE.exists():
        return {"relationships": []}
    with open(RELATIONSHIPS_FILE, encoding="utf-8-sig") as f:
        relationships = json.load(f)
    return {"relationships": [r for r in relationships if r["duration_seconds"] >= min_duration]}


@app.get("/incident")
def get_incident_report(
    start_time: float = Query(...), end_time: float = Query(...),
    min_person_duration: float = Query(1.0),
):
    observation_files = [str(f) for f in discover_observation_files()]
    if not observation_files:
        raise HTTPException(404, "No observation data found")
    if not MATCHES_FILE.exists():
        raise HTTPException(404, f"Matches file not found: {MATCHES_FILE}")
    relationships_path = str(RELATIONSHIPS_FILE) if RELATIONSHIPS_FILE.exists() else None
    events = build_incident_timeline(observation_files, str(MATCHES_FILE), relationships_path,
                                      start_time, end_time)
    narrative = narrate_timeline_template(events, start_time, end_time,
                                           min_person_duration=min_person_duration)
    return {"start_time": start_time, "end_time": end_time, "num_events": len(events),
            "events": events, "narrative": narrative}


@app.get("/search/text")
def search_by_text(query: str = Query(...), top_k: int = Query(5, ge=1, le=50)):
    if not CLIP_EMBEDDINGS_DIR.exists():
        raise HTTPException(404, "No CLIP embeddings found -- run extract_clip_embeddings.py first")
    embedding_files = sorted(CLIP_EMBEDDINGS_DIR.glob("*_clip_embeddings.json"))
    if not embedding_files:
        raise HTTPException(404, "No CLIP embedding files found")
    from clip_search import search as clip_search_fn
    results = clip_search_fn(embedding_paths=[str(f) for f in embedding_files],
                              query=query, top_k=top_k)
    return {"query": query, "results": results}


class VoiceQueryRequest(BaseModel):
    transcript: str


@app.post("/search/voice")
def search_by_voice_transcript(request: VoiceQueryRequest, top_k: int = 5):
    return search_by_text(query=request.transcript, top_k=top_k)


# --- Optional feature modules -------------------------------------------------
# Each routes_*.py file exposes a `router` (a FastAPI APIRouter). If the file
# isn't present, it's silently skipped. If it IS present but broken (e.g. a
# missing dependency), the problem is printed at startup instead of hidden.
OPTIONAL_ROUTE_MODULES = [
       "routes_health", "routes_vehicles", "routes_attributes",
       "routes_qa", "routes_floorplan", "routes_web",
   ]
LOADED_MODULES = []
for _mod_name in OPTIONAL_ROUTE_MODULES:
    try:
        _mod = importlib.import_module(_mod_name)
        app.include_router(_mod.router)
        LOADED_MODULES.append(_mod_name)
    except ModuleNotFoundError as e:
        if e.name != _mod_name:
            print(f"[CatchCam] WARNING: {_mod_name} found but failed to load "
                  f"(missing dependency: {e.name})")
    except Exception as e:
        print(f"[CatchCam] WARNING: {_mod_name} failed to load: {e}")
if LOADED_MODULES:
    print(f"[CatchCam] Loaded optional modules: {', '.join(LOADED_MODULES)}")


@app.get("/modules")
def list_loaded_modules():
    """Which optional feature modules were successfully loaded."""
    return {"loaded": LOADED_MODULES}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
