"""
Remove false detections ("ghost tracks") from a camera's results.

On real footage a detector produces many blips: a "person" that appears for 1-5 frames at low confidence and never moves
(a reflection, a poster, a coat on a chair). Each one gets its own track number, so search results and the track list fill up
with people who were never there. Real people last longer, are more confident, or move.

A track is dropped when ANY of these is true (counted in frames of the real video, so skipping frames does not change the result):
  * it lasts under ~0.25 s                                  (min_frames = 6 at 25 fps)
  * it is low-confidence (mean < 0.5) and shorter than ~2.4 s
  * it is short (< 1 s), almost still and not confident     (a flicker on one spot)
Long, confident tracks are always kept, including people who sit still for minutes.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

from webapp import core

DEFAULTS = dict(min_frames=6, min_conf=0.5, low_conf_frames=60, short_frames=25, still_ratio=0.15, still_conf=0.6)


def _stride(by_track) -> int:
    """How many video frames apart the saved detections are (1 = every frame, 3 = every 3rd...)."""
    gaps = []
    for rows in by_track.values():
        fr = sorted(r["frame"] for r in rows)
        gaps += [b - a for a, b in zip(fr, fr[1:]) if b > a][:50]
    return max(1, int(statistics.median(gaps))) if gaps else 1


def ghost_tracks(rows, **overrides) -> set:
    """Track ids in `rows` (one camera's observations) that look like false detections."""
    p = {**DEFAULTS, **overrides}
    by = defaultdict(list)
    for r in rows:
        by[r["track_id"]].append(r)
    stride = _stride(by)
    ghosts = set()
    for tid, rs in by.items():
        n = len(rs) * stride                                        # frames of video this track covers (approximately)
        conf = sum(r.get("confidence", 1.0) for r in rs) / len(rs)
        boxes = [r["bbox_xyxy"] for r in rs]
        height = sum(b[3] - b[1] for b in boxes) / len(boxes)
        cx = [(b[0] + b[2]) / 2 for b in boxes]
        cy = [(b[1] + b[3]) / 2 for b in boxes]
        moved = ((max(cx) - min(cx)) ** 2 + (max(cy) - min(cy)) ** 2) ** 0.5
        if (n < p["min_frames"]
                or (conf < p["min_conf"] and n < p["low_conf_frames"])
                or (n < p["short_frames"] and moved < p["still_ratio"] * max(height, 1) and conf < p["still_conf"])):
            ghosts.add(tid)
    return ghosts


def _filter_list(path: Path, keep) -> int:
    data = core._read_json(path, None)
    if not isinstance(data, list):
        return 0
    kept = [x for x in data if keep(x)]
    if len(kept) != len(data):
        core._write_json(path, kept)
    return len(data) - len(kept)


def clean_camera(data_dir, cam: str, **overrides) -> dict:
    """Drop ghost tracks from the camera's observations and from everything that refers to them. Returns the counts.
    The original observations are kept once as <cam>_observations.before_cleanup.json."""
    data_dir = Path(data_dir)
    obs_file = data_dir / "outputs" / f"{cam}_observations.json"
    rows = core._read_json(obs_file, [])
    people = [r for r in rows if r.get("class", "person") == "person"]
    ids = {r["track_id"] for r in people}
    ghosts = ghost_tracks(people, **overrides)
    result = {"camera_id": cam, "tracks_before": len(ids), "tracks_after": len(ids) - len(ghosts), "removed_ids": sorted(ghosts)}
    if not ghosts:
        return result
    backup = obs_file.with_name(f"{cam}_observations.before_cleanup.json")
    if not backup.exists():
        backup.write_text(obs_file.read_text(encoding="utf-8-sig"), encoding="utf-8")
    core._write_json(obs_file, [r for r in rows if not (r.get("class", "person") == "person" and r["track_id"] in ghosts)])
    _filter_list(data_dir / "outputs" / f"{cam}_attributes.json", lambda r: r.get("track_id") not in ghosts)
    _filter_list(data_dir / "embeddings" / f"{cam}_embeddings.json", lambda r: not isinstance(r, dict) or r.get("track_id") not in ghosts)
    _filter_list(data_dir / "matches" / "cross_camera_matches.json",
                 lambda m: not ((m.get("camera_a") == cam and m.get("track_a") in ghosts) or (m.get("camera_b") == cam and m.get("track_b") in ghosts)))
    _filter_list(data_dir / "relationships" / "relationships.json",
                 lambda r: not (r.get("camera_id") == cam and (r.get("track_a") in ghosts or r.get("track_b") in ghosts)))
    thumbs = data_dir / ".thumbs"
    for t in ghosts:
        (thumbs / f"{cam}_{t}.jpg").unlink(missing_ok=True)
    return result
