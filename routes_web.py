"""
CatchCam — web interface routes (auto-loaded by api.py as `routes_web`).

  /                       -> redirects to the new interface at /app/
  /app/...                the interface itself (the web/ folder)
  /web/find               the single search box (clothing, plate, track id, or a question)
  /web/follow             best candidates for the same person on every other camera
  /web/subject            everything about one person: journey, gaps, companions, clothing, clips
  /web/cameras            cameras, footage and live status      /web/overview   numbers for the Insights screen
  /video/<camera>         the footage (supports seeking)        /stream/<camera>  server-drawn video with boxes, any codec, live feeds
  /web/frame /web/thumb   still pictures                        /web/boxes       person boxes for the on-video overlay
  /web/upload, /web/process, /web/convert, /web/jobs, /web/streams    adding footage and live feeds

All logic lives in webapp/ (plain Python); this file is only the HTTP layer.
"""
import mimetypes
import os
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse

from webapp import core, jobs

router = APIRouter(tags=["web"])
DATA_DIR = Path(os.environ.get("CATCHCAM_DATA_DIR", "."))
WEB_DIR = Path(__file__).resolve().parent / "web"
NO_CACHE = {"Cache-Control": "no-cache"}


# ----------------------------------------------------------------------------- the interface itself
@router.get("/")
def home():
    return RedirectResponse("/app/")


@router.get("/app")
def app_no_slash():
    return RedirectResponse("/app/")


@router.get("/app/{path:path}")
def app_files(path: str):
    root = WEB_DIR.resolve()
    f = (root / (path or "index.html")).resolve()
    if f != root and root not in f.parents:
        raise HTTPException(403, "Not allowed")
    if not f.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(str(f), headers=NO_CACHE)


# ----------------------------------------------------------------------------- data
@router.get("/web/cameras")
def cameras():
    return {"cameras": core.cameras_overview(DATA_DIR)}


@router.post("/web/cameras/{camera_id}/start")
def camera_start(camera_id: str, seconds: float = Query(..., ge=0, le=10 * 86400)):
    """Set when this camera's video starts on the shared clock, so journeys across cameras line up. No re-processing needed."""
    try:
        cam = core.safe_camera_id(camera_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if cam not in {c["camera_id"] for c in core.cameras_overview(DATA_DIR)}:
        raise HTTPException(404, f"No camera called {cam}.")
    return core.set_camera_start(DATA_DIR, cam, seconds)


@router.post("/web/clean")
def clean(camera_id: str = Query("")):
    """Remove false detections from one camera, or from all of them when camera_id is empty. Works on footage already processed."""
    from webapp import cleaning
    cams = [c["camera_id"] for c in core.cameras_overview(DATA_DIR)]
    if camera_id:
        try:
            camera_id = core.safe_camera_id(camera_id)
        except ValueError as e:
            raise HTTPException(422, str(e))
        if camera_id not in cams:
            raise HTTPException(404, f"No camera called {camera_id}.")
        cams = [camera_id]
    out = [cleaning.clean_camera(DATA_DIR, c) for c in cams]
    return {"cameras": [{k: v for k, v in r.items() if k != "removed_ids"} | {"removed": len(r["removed_ids"])} for r in out],
            "removed": sum(len(r["removed_ids"]) for r in out)}


@router.delete("/web/cameras/{camera_id}")
def delete_camera(camera_id: str):
    """Remove a camera and everything CatchCam stored for it. Refused while it is being processed."""
    try:
        cam = core.safe_camera_id(camera_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if any(j.state in ("queued", "running") and j.camera_id == cam for j in jobs.MANAGER.jobs.values()):
        raise HTTPException(409, f"{cam} is being processed right now. Wait for it to finish, then delete it.")
    known = {c["camera_id"] for c in core.cameras_overview(DATA_DIR)}
    if cam not in known:
        raise HTTPException(404, f"No camera called {cam}.")
    return core.delete_camera(DATA_DIR, cam)


@router.get("/web/overview")
def overview(max_normal_gap: float = Query(15.0, ge=0)):
    return core.overview(DATA_DIR, max_normal_gap)


@router.get("/web/find")
def find(q: str = Query(..., description="Anything: 'person in a red jacket', 'MH12AB1234', 'track 5 on cam1', or a question"),
         context_camera: Optional[str] = Query(None), context_track: Optional[int] = Query(None), context_plate: Optional[str] = Query(None),
         max_normal_gap: float = Query(15.0, ge=0)):
    ctx = None
    if context_plate:
        ctx = {"plate": context_plate}
    elif context_camera is not None and context_track is not None:
        ctx = {"camera_id": context_camera, "track_id": context_track}
    return core.find(DATA_DIR, q, ctx, max_normal_gap)


@router.get("/web/subject")
def subject(camera_id: str = Query(...), track_id: int = Query(...), max_normal_gap: float = Query(15.0, ge=0),
            critical_gap: float = Query(45.0, ge=0)):
    try:
        return core.subject_bundle(DATA_DIR, camera_id, track_id, max_normal_gap, critical_gap)
    except LookupError as e:
        raise HTTPException(404, str(e))


@router.get("/web/follow")
def follow(camera_id: str = Query(...), track_id: int = Query(...), per_camera: int = Query(3, ge=1, le=8)):
    """Best candidates for 'the same person' on every other camera, ranked, for a human to confirm."""
    from webapp import follow as _follow
    try:
        return _follow.follow(DATA_DIR, camera_id, track_id, per_camera)
    except LookupError as e:
        raise HTTPException(404, str(e))


@router.get("/web/boxes")
def boxes(camera_id: str = Query(...), start: float = Query(0.0), end: float = Query(60.0), track_id: Optional[int] = Query(None)):
    """Rows [t, track, x1, y1, x2, y2] on the shared clock, for drawing boxes over the video."""
    if end - start > 600:
        raise HTTPException(422, "Ask for at most 10 minutes at a time.")
    return {"camera_id": camera_id, "start": start, "end": end, "rows": core.boxes_window(DATA_DIR, camera_id, start, end, track_id)}


@router.get("/web/frame")
def frame(camera_id: str = Query(...), t: float = Query(0.0), track_id: Optional[int] = Query(None), width: int = Query(960, ge=160, le=1920)):
    jpg = core.frame_jpeg(DATA_DIR, camera_id, t, track_id, True, width)
    if jpg is None:
        raise HTTPException(404, f"No footage for {camera_id} at {t:.1f}s")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "max-age=60"})


@router.get("/web/thumb")
def thumb(camera_id: str = Query(...), track_id: int = Query(...)):
    jpg = core.thumb_jpeg(DATA_DIR, camera_id, track_id)
    if jpg is None:
        raise HTTPException(404, "No picture available")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "max-age=300"})


# ----------------------------------------------------------------------------- video
@router.get("/video/{camera_id}")
def video(camera_id: str, request: Request):
    path, _ = core.playable_video(DATA_DIR, camera_id)
    if path is None:
        raise HTTPException(404, f"No footage for {camera_id}")
    size = path.stat().st_size
    rng = core.parse_range(request.headers.get("range"), size)
    start, end = rng or (0, size - 1)

    def chunks():
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                b = f.read(min(1 << 20, left))
                if not b:
                    break
                left -= len(b)
                yield b

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(end - start + 1), "Cache-Control": "no-cache"}
    if rng:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(chunks(), status_code=206 if rng else 200,
                             media_type=mimetypes.guess_type(str(path))[0] or "video/mp4", headers=headers)


@router.get("/stream/{camera_id}")
def stream(camera_id: str, start: float = Query(0.0), end: Optional[float] = Query(None), track_id: Optional[int] = Query(None),
           speed: float = Query(1.0, ge=0.1, le=8.0), loop: int = Query(0), width: int = Query(960, ge=160, le=1920)):
    if camera_id not in core.camera_videos(DATA_DIR) and camera_id not in core.camera_streams(DATA_DIR):
        raise HTTPException(404, f"No footage or live source for {camera_id}")
    gen = core.mjpeg(DATA_DIR, camera_id, start, end, track_id, speed, bool(loop), width)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------------------------- adding footage
@router.post("/web/upload")
async def upload(request: Request, camera_id: str = Query(...), filename: str = Query("video.mp4")):
    """The browser sends the video file as the raw request body (no extra packages needed)."""
    try:
        cam = core.safe_camera_id(camera_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    ext = Path(filename).suffix.lower()
    if ext not in core.VIDEO_EXT:
        raise HTTPException(422, f"Can't use a {ext or 'file without an extension'}. Use mp4, mov, webm, avi or mkv.")
    dest = DATA_DIR / "videos" / f"{cam}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part, size = dest.with_suffix(".part"), 0
    with open(part, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
            size += len(chunk)
    if size == 0:
        part.unlink(missing_ok=True)
        raise HTTPException(422, "The file was empty.")
    part.replace(dest)
    cap = cv2.VideoCapture(str(dest))
    readable = cap.isOpened()
    cap.release()
    if not readable:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "This video can't be read. Try an MP4 (H.264) file.")
    core.register_video(DATA_DIR, cam, dest)
    _, playable = core.playable_video(DATA_DIR, cam)
    return {"camera_id": cam, "size_mb": round(size / 1e6, 1), "plays_in_browser": playable}


@router.post("/web/process")
def process(camera_id: str = Query(...), time_offset: float = Query(0.0), weights: str = Query("yolov8n.pt"),
            speed: str = Query("normal"), sensitivity: str = Query("normal"), skip_empty: int = Query(1)):
    if speed not in jobs.SPEEDS or sensitivity not in jobs.SENSITIVITY:
        raise HTTPException(422, f"speed must be one of {list(jobs.SPEEDS)} and sensitivity one of {list(jobs.SENSITIVITY)}.")
    if camera_id not in core.camera_videos(DATA_DIR):
        raise HTTPException(404, f"No footage for {camera_id}. Upload a video first.")
    w = weights
    if not Path(w).is_absolute() and (DATA_DIR / w).exists():
        w = str(DATA_DIR / w)
    try:
        job = jobs.MANAGER.start("process", camera_id, lambda j: jobs.process_footage(DATA_DIR, camera_id, j, w, time_offset, speed, sensitivity, bool(skip_empty)))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return job.to_dict()


@router.post("/web/convert")
def convert(camera_id: str = Query(...)):
    if camera_id not in core.camera_videos(DATA_DIR):
        raise HTTPException(404, f"No footage for {camera_id}")
    try:
        job = jobs.MANAGER.start("convert", camera_id, lambda j: jobs.convert_only(DATA_DIR, camera_id, j))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return job.to_dict()


@router.get("/web/jobs")
def list_jobs():
    return {"jobs": [j.to_dict() for j in jobs.MANAGER.recent()]}


@router.get("/web/jobs/{job_id}")
def get_job(job_id: str):
    j = jobs.MANAGER.get(job_id)
    if j is None:
        raise HTTPException(404, "No such job")
    return j.to_dict()


@router.post("/web/streams")
def add_stream(camera_id: str = Query(...), url: str = Query(..., description="rtsp://… or http://… address, or a webcam number like 0")):
    try:
        cam = core.safe_camera_id(camera_id)
    except ValueError as e:
        raise HTTPException(422, str(e))
    cap = cv2.VideoCapture(int(url) if url.isdigit() else url)
    ok = cap.isOpened()
    cap.release()
    if not ok:
        raise HTTPException(422, "Couldn't open that address. Check the URL, username/password and that the camera is on the same network.")
    core.register_stream(DATA_DIR, cam, url)
    return {"camera_id": cam, "url": url}


# ----------------------------------------------------------------------------- extra tools: loaded from here so api.py needs no change
# (each one is skipped quietly when its file is not in the folder; a file that is there but broken is reported at startup)
import importlib  # noqa: E402

for _name in ("routes_eval", "routes_report", "routes_alerts"):
    try:
        router.include_router(importlib.import_module(_name).router)
    except ModuleNotFoundError as _e:
        if _e.name != _name:
            print(f"[CatchCam] WARNING: {_name} found but failed to load (missing dependency: {_e.name})")
    except Exception as _e:                                                           # noqa: BLE001
        print(f"[CatchCam] WARNING: {_name} failed to load: {_e}")
