"""
CatchCam web interface — backend logic (plain Python, no FastAPI), so it can be tested without a server.

  cameras_overview   every camera: footage, status, how many people were tracked
  subject_bundle     everything the Investigate screen needs about ONE person, in one call
  find               the single search box: clothing, plate, track id, or a spoken/typed question
  frame_jpeg / thumb_jpeg / mjpeg   pictures and video streams drawn from the footage
  overview           numbers and flagged journeys for the Insights screen
"""
import bisect
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from build_graph import node_id
from missing_camera import check_live_camera_health, find_coverage_gaps, get_full_journey
from person_attrs.core import match_agreement, matches_all, parse_description, search_people
from qa_engine.engine import EXAMPLES, Store, ask, resolve_subject

SUBJECT_BGR = (63, 210, 255)            # #ffd23f — the one colour that always means "the person being followed"
OTHER_BGR = (190, 165, 130)             # everyone else
BROWSER_CODECS = {"avc1", "h264", "vp80", "vp90", "av01"}
BROWSER_EXT = {".mp4", ".m4v", ".webm", ".mov"}
VIDEO_EXT = {".mp4", ".m4v", ".webm", ".mov", ".avi", ".mkv"}


# ----------------------------------------------------------------------------- small helpers
def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def _write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def safe_camera_id(name: str) -> str:
    cid = re.sub(r"[^A-Za-z0-9_\-]", "", (name or "").strip().replace(" ", "_"))
    if not cid:
        raise ValueError("Camera name can only contain letters, numbers, - and _.")
    return cid[:40]


# ----------------------------------------------------------------------------- footage registry
def camera_videos(data_dir) -> dict:
    """camera_id -> existing video file, from camera_videos.json (the file the rest of CatchCam already uses)."""
    data_dir = Path(data_dir)
    out = {}
    for cam, p in _read_json(data_dir / "camera_videos.json", {}).items():
        cand = Path(p) if Path(p).is_absolute() else data_dir / p
        if not cand.exists() and Path(p).exists():
            cand = Path(p)
        if cand.exists():
            out[cam] = cand
    return out


def register_video(data_dir, cam: str, path) -> None:
    data_dir = Path(data_dir)
    m = _read_json(data_dir / "camera_videos.json", {})
    p = Path(path)
    try:
        p = p.resolve().relative_to(data_dir.resolve())
    except ValueError:
        pass
    m[cam] = p.as_posix()
    _write_json(data_dir / "camera_videos.json", m)


def camera_streams(data_dir) -> dict:
    return _read_json(Path(data_dir) / "streams.json", {})


def register_stream(data_dir, cam: str, source: str) -> None:
    m = camera_streams(data_dir)
    m[cam] = source
    _write_json(Path(data_dir) / "streams.json", m)


def set_camera_start(data_dir, cam: str, start: float) -> dict:
    """Say when this camera's video starts on the shared clock (seconds). Everything already found in it is shifted to match,
    so footage does not need to be processed again. Use the on-screen clock: 12:38:53 -> 45533."""
    data_dir = Path(data_dir)
    old = _t0(data_dir, cam)
    delta = float(start) - old
    shifted = 0
    keys = ("timestamp", "first_seen_timestamp", "last_seen_timestamp")
    out = data_dir / "outputs"
    for f in (out.glob(f"{cam}_*.json") if out.is_dir() else []):
        rows = _read_json(f, None)
        if not isinstance(rows, list) or not any(isinstance(r, dict) and any(k in r for k in keys) for r in rows):
            continue
        for r in rows:
            if isinstance(r, dict):
                for k in keys:
                    if isinstance(r.get(k), (int, float)):
                        r[k] = round(r[k] + delta, 3)
        _write_json(f, rows)
        shifted += 1
    offs = _read_json(data_dir / "time_offsets.json", {})
    offs[cam] = float(start)
    _write_json(data_dir / "time_offsets.json", offs)
    _CODEC.clear()
    return {"camera_id": cam, "start": float(start), "shifted_by": round(delta, 3), "files_updated": shifted}


def delete_camera(data_dir, cam: str) -> dict:
    """Remove one camera: its video, converted copy, tracking/clothing/appearance files, its matches and its registry entries."""
    data_dir = Path(data_dir)
    removed = []

    def rm(path):
        path = Path(path)
        if path.is_file():
            path.unlink()
            removed.append(path.name)

    videos = camera_videos(data_dir)
    if cam in videos:
        rm(videos[cam])
    for p in (data_dir / "videos").glob(f"{cam}.*") if (data_dir / "videos").is_dir() else []:
        rm(p)
    for sub in ("outputs", "embeddings"):
        d = data_dir / sub
        for p in (d.glob(f"{cam}_*") if d.is_dir() else []):
            rm(p)
    for name in ("camera_videos.json", "streams.json", "time_offsets.json"):
        f = data_dir / name
        m = _read_json(f, None)
        if isinstance(m, dict) and cam in m:
            m.pop(cam)
            _write_json(f, m)
    mf = data_dir / "matches" / "cross_camera_matches.json"
    ms = _read_json(mf, None)
    if isinstance(ms, list):
        keep = [x for x in ms if cam not in (x.get("camera_a"), x.get("camera_b"))]
        if len(keep) != len(ms):
            _write_json(mf, keep)
    _CODEC.clear()
    return {"camera_id": cam, "removed_files": removed}


def web_copy_path(data_dir, cam) -> Path:
    return Path(data_dir) / "videos" / f"{cam}.web.mp4"


_CODEC = {}


def video_codec(path) -> str:
    key = (str(path), Path(path).stat().st_mtime)
    if key not in _CODEC:
        cap = cv2.VideoCapture(str(path))
        f = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        cap.release()
        _CODEC[key] = "".join(chr((f >> 8 * i) & 0xFF) for i in range(4)).strip().lower() if f else ""
    return _CODEC[key]


def playable_video(data_dir, cam):
    """(path, plays_in_browser). Prefers the converted copy; OpenCV's default 'mp4v' files do NOT play in browsers."""
    web = web_copy_path(data_dir, cam)
    src = camera_videos(data_dir).get(cam)
    if web.exists() and (not src or web.stat().st_mtime >= src.stat().st_mtime):
        return web, True
    if not src:
        return None, False
    return src, src.suffix.lower() in BROWSER_EXT and video_codec(src) in BROWSER_CODECS


def make_web_copy(data_dir, cam, progress=None) -> bool:
    """Convert to H.264 MP4 with ffmpeg (if installed). Returns False when ffmpeg is missing."""
    src = camera_videos(data_dir).get(cam)
    ff = shutil.which("ffmpeg")
    if not src or not ff:
        return False
    dst = web_copy_path(data_dir, cam)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [ff, "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg could not convert this video: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1])
    tmp.replace(dst)
    return True


# ----------------------------------------------------------------------------- observations
_OBS = {}


def obs_index(data_dir, cam):
    """(rows sorted by time, parallel list of timestamps) for one camera; cached until the file changes."""
    f = Path(data_dir) / "outputs" / f"{cam}_observations.json"
    if not f.exists():
        return [], []
    m = f.stat().st_mtime
    hit = _OBS.get(str(f))
    if hit and hit[0] == m:
        return hit[1], hit[2]
    rows = sorted(_read_json(f, []), key=lambda r: (r["timestamp"], r["track_id"]))
    idx = [r["timestamp"] for r in rows]
    _OBS[str(f)] = (m, rows, idx)
    return rows, idx


def boxes_window(data_dir, cam, start: float, end: float, track_id: Optional[int] = None, limit: int = 20000):
    """Compact rows [t, track, x1, y1, x2, y2] for the overlay on the video."""
    rows, idx = obs_index(data_dir, cam)
    lo, hi = bisect.bisect_left(idx, start), bisect.bisect_right(idx, end)
    out = []
    for r in rows[lo:hi]:
        if track_id is not None and r["track_id"] != track_id:
            continue
        x1, y1, x2, y2 = r["bbox_xyxy"]
        out.append([r["timestamp"], r["track_id"], round(x1), round(y1), round(x2), round(y2)])
        if len(out) >= limit:
            break
    return out


def known_camera_ids(data_dir):
    d = Path(data_dir)
    ids = {f.name[: -len("_observations.json")] for f in (d / "outputs").glob("*_observations.json")} if (d / "outputs").exists() else set()
    ids |= set(camera_videos(d)) | set(camera_streams(d))
    if (d / "outputs").exists():
        ids |= {f.name[: -len("_observations_live.json")] for f in (d / "outputs").glob("*_observations_live.json")}
    return sorted(ids)


# ----------------------------------------------------------------------------- cameras
def cameras_overview(data_dir):
    d = Path(data_dir)
    videos, streams = camera_videos(d), camera_streams(d)
    health = {c["camera_id"]: c for c in check_live_camera_health(str(d / "outputs"), 10.0)["cameras"]}
    out = []
    for cam in known_camera_ids(d):
        rows, idx = obs_index(d, cam)
        path, playable = playable_video(d, cam)
        if cam in health:
            h = health[cam]
            status = "down" if h["status"].startswith("DOWN") else "live"
            ago = h["last_update_seconds_ago"]
        else:
            status, ago = ("recorded" if cam in videos else ("stream" if cam in streams else "no_footage")), None
        out.append({
            "camera_id": cam, "status": status, "last_update_seconds_ago": ago,
            "has_video": path is not None, "plays_in_browser": playable, "can_convert": bool(path and not playable and shutil.which("ffmpeg")),
            "is_stream": cam in streams, "offset": _t0(d, cam), "video_url": f"/video/{cam}" if path else None,
            "stream_url": f"/stream/{cam}" if (path or cam in streams) else None,
            "num_tracks": len({r["track_id"] for r in rows}), "first_seen": idx[0] if idx else None, "last_seen": idx[-1] if idx else None,
            "duration": _video_duration(path) if path else None,
        })
    return out


def _video_duration(path) -> Optional[float]:
    cap = cv2.VideoCapture(str(path))
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return round(n / fps, 2) if fps and n else None


# ----------------------------------------------------------------------------- pictures
def _draw(frame, boxes, subject_track=None, labels=True):
    for (tid, x1, y1, x2, y2) in boxes:
        mine = subject_track is not None and tid == subject_track
        col, th = (SUBJECT_BGR, 3) if mine else (OTHER_BGR, 1 if subject_track is not None else 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, th)
        if labels:
            txt = f"#{tid}"
            (tw, tt), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - tt - 8), (int(x1) + tw + 8, int(y1)), col, -1)
            cv2.putText(frame, txt, (int(x1) + 4, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def _boxes_near(rows, idx, t, tol=0.35):
    lo, hi = bisect.bisect_left(idx, t - tol), bisect.bisect_right(idx, t + tol)
    best = {}
    for r in rows[lo:hi]:
        k = r["track_id"]
        if k not in best or abs(r["timestamp"] - t) < abs(best[k]["timestamp"] - t):
            best[k] = r
    return [(k, *r["bbox_xyxy"]) for k, r in best.items()]


def _jpeg(img, quality=82, width=None):
    if width and img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def read_frame_at(path, t):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def frame_jpeg(data_dir, cam, t, track_id=None, boxes=True, width=960):
    src = camera_videos(data_dir).get(cam)
    if not src:
        return None
    rel = t - _t0(data_dir, cam)                                 # t is on the shared clock; the file starts at its offset
    if rel < -0.05:                                              # before this camera's footage begins: no picture, not the first frame
        return None
    frame = read_frame_at(src, rel)
    if frame is None:
        return None
    if boxes:
        rows, idx = obs_index(data_dir, cam)
        _draw(frame, _boxes_near(rows, idx, t), track_id)
    return _jpeg(frame, width=width)


def thumb_jpeg(data_dir, cam, track_id, height=150):
    """A crop of the person (best-looking detection of the track), cached on disk."""
    cache = Path(data_dir) / ".thumbs" / f"{cam}_{track_id}.jpg"
    src = camera_videos(data_dir).get(cam)
    obs_file = Path(data_dir) / "outputs" / f"{cam}_observations.json"
    newest = max(src.stat().st_mtime if src else 0, obs_file.stat().st_mtime if obs_file.exists() else 0)
    if cache.exists() and src and cache.stat().st_mtime >= newest:      # re-processing changes track ids, so refresh then
        return cache.read_bytes()
    if not src:
        return None
    rows, _ = obs_index(data_dir, cam)
    mine = [r for r in rows if r["track_id"] == track_id]
    if not mine:
        return None
    best = max(mine, key=lambda r: (r["bbox_xyxy"][2] - r["bbox_xyxy"][0]) * (r["bbox_xyxy"][3] - r["bbox_xyxy"][1]) * max(r.get("confidence", 0.5), 0.1))
    frame = read_frame_at(src, best["timestamp"] - _t0(data_dir, cam))
    if frame is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in best["bbox_xyxy"]]
    pw, ph = int((x2 - x1) * 0.15), int((y2 - y1) * 0.08)
    crop = frame[max(y1 - ph, 0):min(y2 + ph, frame.shape[0]), max(x1 - pw, 0):min(x2 + pw, frame.shape[1])]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (max(int(crop.shape[1] * height / crop.shape[0]), 1), height), interpolation=cv2.INTER_AREA)
    data = _jpeg(crop, 85)
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(data)
    return data


def _t0(data_dir, cam) -> float:
    """Time (on the shared clock) at which this camera's video starts: 0 unless footage was processed with an offset."""
    return float(_read_json(Path(data_dir) / "time_offsets.json", {}).get(cam, 0.0))


def mjpeg(data_dir, cam, start=0.0, end=None, track_id=None, speed=1.0, loop=False, width=960, stop=lambda: False):
    """Yields multipart/x-mixed-replace chunks. Works for ANY codec OpenCV can read and for live streams."""
    d = Path(data_dir)
    live_src = camera_streams(d).get(cam)
    src = live_src if live_src is not None else camera_videos(d).get(cam)
    if src is None:
        return
    is_live = live_src is not None
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else str(src))
    if not cap.isOpened():
        return
    fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) if not is_live else 0
    rows, idx = obs_index(d, cam)
    off = _t0(d, cam)
    start = max(start - off, 0.0)                              # start / end arrive on the shared clock
    end = None if end is None else end - off
    if not is_live and start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    frame_dt = (1.0 / fps) / max(speed, 0.1) if fps else 0
    nxt, failures = time.time(), 0
    try:
        while not stop():
            ok, frame = cap.read()
            if not ok:
                if is_live and failures < 20:                     # camera hiccup: try again, then reconnect
                    failures += 1
                    time.sleep(0.25)
                    cap.release()
                    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else str(src))
                    continue
                if loop and not is_live:
                    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
                    nxt = time.time()
                    continue
                break
            failures = 0
            t = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000.0
            if end is not None and t > end:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
                    nxt = time.time()
                    continue
                break
            if not is_live and idx:
                _draw(frame, _boxes_near(rows, idx, t + off), track_id)
            jpg = _jpeg(frame, 78, width)
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            if frame_dt:
                nxt += frame_dt
                time.sleep(max(0.0, nxt - time.time()))
    finally:
        cap.release()


def parse_range(header: Optional[str], size: int):
    """'bytes=100-' -> (100, size-1); None if absent/invalid. Lets the browser seek inside a video."""
    if not header or not header.startswith("bytes="):
        return None
    a, _, b = header[6:].split(",")[0].partition("-")
    try:
        if a == "":
            n = int(b)
            return (max(size - n, 0), size - 1) if n > 0 else None
        start = int(a)
        end = int(b) if b else size - 1
    except ValueError:
        return None
    end = min(end, size - 1)
    return (start, end) if 0 <= start <= end else None


# ----------------------------------------------------------------------------- one person, everything
def _clothing(row):
    if not row:
        return None
    return {"upper": row.get("upper_color"), "upper_conf": row.get("upper_conf"),
            "lower": row.get("lower_color"), "lower_conf": row.get("lower_conf")}


def _thumb(cam, tid):
    return f"/web/thumb?camera_id={cam}&track_id={tid}"


def subject_bundle(data_dir, camera_id, track_id, max_gap=15.0, critical_gap=45.0):
    d = Path(data_dir)
    store = Store(d)
    G = store.graph()
    if G is None:
        raise LookupError("There is no tracking and matching data yet. Add footage in Cameras, or run the pipeline scripts.")
    if node_id(camera_id, track_id) not in G.nodes:
        raise LookupError(f"No sighting found for {camera_id} track {track_id}.")
    journey = get_full_journey(G, camera_id, track_id)
    gaps = find_coverage_gaps(journey, max_gap)
    for g in gaps:
        g["severity"] = "critical" if g["gap_seconds"] > critical_gap else "suspicious"
    attrs = {(r["camera_id"], r["track_id"]): r for r in store.attributes()}
    videos = camera_videos(d)
    hops, prev = [], None
    for h in journey:
        cam, tid = h["camera_id"], h["track_id"]
        sim, agreement = None, None
        if prev and G.has_edge(node_id(prev["camera_id"], prev["track_id"]), node_id(cam, tid)):
            sim = G[node_id(prev["camera_id"], prev["track_id"])][node_id(cam, tid)]["similarity"]
            agreement = match_agreement(attrs.get((prev["camera_id"], prev["track_id"])), attrs.get((cam, tid)))
        hops.append({"camera_id": cam, "track_id": tid, "start": h["first_seen_timestamp"], "end": h["last_seen_timestamp"],
                     "duration": round(h["last_seen_timestamp"] - h["first_seen_timestamp"], 2), "similarity": sim,
                     "clothing_check": agreement, "clothing": _clothing(attrs.get((cam, tid))),
                     "is_selected": (cam, tid) == (camera_id, track_id), "has_video": cam in videos, "thumb_url": _thumb(cam, tid)})
        prev = h
    chain = {(h["camera_id"], h["track_id"]) for h in hops}
    companions = []
    for r in store.relationships():
        for (cam, tid) in chain:
            if r["camera_id"] == cam and tid in (r["track_a"], r["track_b"]):
                other = r["track_b"] if r["track_a"] == tid else r["track_a"]
                companions.append({"camera_id": cam, "track_id": other, "with_track": tid, "start": r["start_timestamp"],
                                   "end": r["end_timestamp"], "duration": r["duration_seconds"],
                                   "clothing": _clothing(attrs.get((cam, other))), "thumb_url": _thumb(cam, other)})
    companions.sort(key=lambda c: c["start"])
    narrative = ask(f"show the journey of track {track_id} on {camera_id}", store, max_gap)
    starts, ends = [h["start"] for h in hops], [h["end"] for h in hops]
    return {"camera_id": camera_id, "track_id": track_id, "hops": hops, "gaps": gaps, "has_blind_spot": bool(gaps),
            "companions": companions, "narrative": narrative["answer"], "speak": narrative["speak"],
            "cameras": sorted(set(known_camera_ids(d)) | {h["camera_id"] for h in hops}),
            "t_min": min(starts), "t_max": max(ends), "max_normal_gap": max_gap,
            "floorplan_url": (f"/floorplan/render?camera_id={camera_id}&track_id={track_id}&max_normal_gap={max_gap}"
                              if (d / "floorplan.json").exists() else None)}


# ----------------------------------------------------------------------------- the single search box
_QUESTION = re.compile(r"^(where|who|when|what|which|how|is|are|did|does|was|were|any|tell|show|find|follow)\b")


def _person_card(row, score=None):
    return {"kind": "person", "camera_id": row["camera_id"], "track_id": row["track_id"],
            "title": f"{row['camera_id']} · track {row['track_id']}", "first_seen": row.get("first_seen_timestamp"),
            "last_seen": row.get("last_seen_timestamp"), "clothing": _clothing(row), "score": score,
            "thumb_url": _thumb(row["camera_id"], row["track_id"])}


def find(data_dir, q: str, context: Optional[dict] = None, max_gap: float = 15.0):
    """
    One entry point for everything typed or spoken. Returns:
      answer / speak   a sentence built from the data (None if the text wasn't a question we understand)
      results          people or vehicles to pick from (with thumbnails)
      selected         the person/vehicle to open straight away, if it's clear who is meant
      context          send back with the next question so "he" / "that car" work
    """
    d = Path(data_dir)
    store = Store(d)
    text = re.sub(r"\s+", " ", (q or "").strip())
    if not text:
        return {"query": q, "answer": None, "speak": "", "results": [], "selected": None, "context": context, "examples": EXAMPLES[:4]}
    low = text.lower()
    res = ask(text, store, max_gap, context)
    understood = res["intent"] != "unknown"
    results, selected = [], None

    subj = resolve_subject(low, store)
    want = parse_description(low)
    if subj.kind in ("plate", "vehicles"):
        for r in (subj.vehicle_rows or [])[:12]:
            results.append({"kind": "vehicle", "camera_id": r["camera_id"], "track_id": r["track_id"], "plate": r.get("plate"),
                            "title": " ".join(x for x in (r.get("color"), r.get("vehicle_type")) if x) or "vehicle",
                            "first_seen": r.get("first_seen_timestamp"), "last_seen": r.get("last_seen_timestamp"),
                            "thumb_url": _thumb(r["camera_id"], r["track_id"])})
        if subj.kind == "plate" and subj.vehicle_rows:
            selected = {"kind": "vehicle", "plate": subj.plate}
    elif any(want.values()):
        rows = [r for r in search_people(store.attributes(), low, limit=24) if matches_all(r, want)]
        results = [_person_card(r, r["score"]) for r in rows[:12]]
    elif subj.kind == "track" and subj.camera_id is not None:
        G = store.graph()
        if G is not None and node_id(subj.camera_id, subj.track_id) in G.nodes:
            n = G.nodes[node_id(subj.camera_id, subj.track_id)]
            results = [_person_card({"camera_id": subj.camera_id, "track_id": subj.track_id,
                                     "first_seen_timestamp": n["first_seen_timestamp"], "last_seen_timestamp": n["last_seen_timestamp"]})]

    results = _one_card_per_person(store, results)
    ctx = res.get("context")
    if selected is None and ctx and "camera_id" in ctx:
        selected = {"kind": "person", **ctx}
    elif selected is None and results and results[0]["kind"] == "person" and (len(results) == 1 or understood):
        selected = {"kind": "person", "camera_id": results[0]["camera_id"], "track_id": results[0]["track_id"]}

    if not understood and not results:                                   # last resort: look at the pictures themselves
        try:
            results = _clip_results(d, text)
        except Exception:
            results = []
    answer = res["answer"] if (understood or not results) else None
    speak = res["speak"] if answer else ""
    if len(results) == 1:                              # "3 people match" counted the same person on 3 cameras
        note = re.compile(r"\s*\(?\d+ people match; answering for the best match\.\)?")
        answer, speak = (note.sub("", answer) if answer else answer), note.sub("", speak)
    return {"query": q, "answer": answer, "speak": speak, "intent": res["intent"], "results": results,
            "selected": selected, "context": res.get("context"), "examples": EXAMPLES[:4]}


def _one_card_per_person(store, results):
    """The same person seen on three cameras matches a clothing search three times: show one card with their route."""
    G = store.graph()
    if G is None:
        return results
    seen, out = {}, []
    for card in results:
        if card["kind"] != "person" or node_id(card["camera_id"], card["track_id"]) not in G.nodes:
            out.append(card)
            continue
        j = get_full_journey(G, card["camera_id"], card["track_id"])
        key = tuple((h["camera_id"], h["track_id"]) for h in j)
        if key in seen:
            continue
        card["route"] = " → ".join(h["camera_id"] for h in j)
        card["sightings"] = len(j)
        seen[key] = card
        out.append(card)
    return out


def _clip_results(d: Path, text: str, top_k: int = 8):
    emb = d / "clip_embeddings"
    files = sorted(emb.glob("*_clip_embeddings.json")) if emb.exists() else []
    if not files:
        return []
    from clip_search import search as clip_search        # heavy import: only when nothing else understood the text
    out = []
    for r in clip_search(embedding_paths=[str(f) for f in files], query=text, top_k=top_k):
        out.append({"kind": "person", "camera_id": r["camera_id"], "track_id": r["track_id"], "title": f"{r['camera_id']} · track {r['track_id']}",
                    "first_seen": r.get("first_seen_timestamp"), "last_seen": r.get("last_seen_timestamp"), "clothing": None,
                    "score": r["similarity"], "thumb_url": _thumb(r["camera_id"], r["track_id"])})
    return out


# ----------------------------------------------------------------------------- insights
def overview(data_dir, max_gap=15.0, buckets=24):
    d = Path(data_dir)
    store = Store(d)
    cams = cameras_overview(d)
    G = store.graph()
    matches = _read_json(store.matches, []) if store.matches.exists() else []
    t_all = [c["first_seen"] for c in cams if c["first_seen"] is not None] + [c["last_seen"] for c in cams if c["last_seen"] is not None]
    t0, t1 = (min(t_all), max(t_all)) if t_all else (0.0, 0.0)
    span = max(t1 - t0, 1e-6)
    activity = {}
    if G is not None:
        for _, n in G.nodes(data=True):
            b = min(int((n["first_seen_timestamp"] - t0) / span * buckets), buckets - 1)
            activity.setdefault(n["camera_id"], [0] * buckets)[b] += 1
    flagged = []
    if G is not None:
        for n in G.nodes:
            if G.in_degree(n) == 0 and G.out_degree(n) > 0:             # start of a chain
                a = G.nodes[n]
                j = get_full_journey(G, a["camera_id"], a["track_id"])
                gaps = find_coverage_gaps(j, max_gap)
                if gaps:
                    flagged.append({"camera_id": a["camera_id"], "track_id": a["track_id"], "route": " → ".join(h["camera_id"] for h in j),
                                    "longest_gap": max(g["gap_seconds"] for g in gaps), "gaps": len(gaps)})
    flagged.sort(key=lambda f: -f["longest_gap"])
    attrs = {(r["camera_id"], r["track_id"]): r for r in store.attributes()}
    conflicts = [m for m in matches if match_agreement(attrs.get((m["camera_a"], m["track_a"])), attrs.get((m["camera_b"], m["track_b"])))["verdict"] == "conflict"]
    return {"cameras": cams, "people_tracked": sum(c["num_tracks"] for c in cams), "matches": len(matches),
            "with_clothing_data": len(attrs), "vehicles": len(store.vehicles()), "cameras_down": [c["camera_id"] for c in cams if c["status"] == "down"],
            "t_min": t0 if t_all else None, "t_max": t1 if t_all else None, "activity": activity, "buckets": buckets,
            "flagged_journeys": flagged[:12], "match_conflicts": [{"a": f"{m['camera_a']} #{m['track_a']}", "b": f"{m['camera_b']} #{m['track_b']}",
                                                                   "similarity": m["similarity"]} for m in conflicts[:12]]}
