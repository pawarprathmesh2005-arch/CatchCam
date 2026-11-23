"""Follow one person across every camera, without relying on the 0.75 'same person' cut-off.

The saved matches treat anyone with a raw ResNet similarity above 0.75 as the same person, but raw similarities between
*different* people sit around 0.55 +/- 0.12 on real footage, so that cut-off lets nearly everybody through. Here the
average appearance of all tracks is subtracted first (so only what makes a person different is compared), clothing
colours are compared with 'gray/silver/white' treated as one washed-out family, and, when camera start times are known,
candidates that are impossibly far apart in time are dropped. Result: the best few candidates per camera, ranked, for a
human to confirm.
"""
import json
from pathlib import Path

import numpy as np

NEUTRAL = {"gray", "grey", "silver", "white"}
DARK = {"black", "navy"}


def _read(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def colour_score(a, b):
    """1 = same colour, 0.6 = both washed-out neutrals, 0.5 = unknown, 0.35 = dark vs neutral, 0 = different."""
    if not a or not b:
        return 0.5
    a, b = str(a).lower(), str(b).lower()
    if a == b:
        return 1.0
    if a in NEUTRAL and b in NEUTRAL:
        return 0.6
    if (a in DARK and b in NEUTRAL) or (b in DARK and a in NEUTRAL):
        return 0.35
    if a in DARK and b in DARK:
        return 0.9
    return 0.0


def _load(data_dir):
    d = Path(data_dir)
    live, attrs, emb = {}, {}, {}
    for f in sorted((d / "outputs").glob("*_observations.json")):
        cam = f.name[: -len("_observations.json")]
        if ".before_cleanup" in f.name:
            continue
        for r in _read(f, []):
            if isinstance(r, dict) and r.get("class", "person") == "person":
                k = (cam, int(r["track_id"]))
                a, b = live.get(k, (r["timestamp"], r["timestamp"]))
                live[k] = (min(a, r["timestamp"]), max(b, r["timestamp"]))
    for f in (d / "outputs").glob("*_attributes.json"):
        for r in _read(f, []):
            attrs[(r["camera_id"], int(r["track_id"]))] = r
    for f in (d / "embeddings").glob("*_embeddings.json"):
        j = _read(f, {})
        for t, v in (j.get("tracks") or {}).items():
            k = (j.get("camera_id"), int(t))
            if k in live and v.get("embedding"):
                emb[k] = np.asarray(v["embedding"], float)
    return live, attrs, emb


def follow(data_dir, camera_id, track_id, per_camera=3, max_time_gap=900.0):
    d = Path(data_dir)
    live, attrs, emb = _load(d)
    seed = (camera_id, int(track_id))
    if seed not in live:
        raise LookupError(f"No sighting found for {camera_id} track {track_id}.")
    offs = _read(d / "time_offsets.json", {})
    keys = list(emb)
    if seed not in emb:
        raise LookupError("This person has no appearance data yet. Process the camera first (Cameras -> Process).")
    X = np.stack([emb[k] for k in keys])
    X = X - X.mean(0)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(n == 0, 1, n)
    sv = X[keys.index(seed)]
    sims = X @ sv
    sa = attrs.get(seed, {})
    seed_t = live[seed]
    out = {}
    for i, k in enumerate(keys):
        if k[0] == camera_id:
            continue
        if k[0] in offs and camera_id in offs:
            gap = max(live[k][0] - seed_t[1], seed_t[0] - live[k][1], 0.0)
            if gap > max_time_gap:
                continue
        ca = attrs.get(k, {})
        cu, cl = colour_score(sa.get("upper_color"), ca.get("upper_color")), colour_score(sa.get("lower_color"), ca.get("lower_color"))
        app = float(np.clip((sims[i] + 0.1) / 0.6, 0, 1))
        score = 0.55 * app + 0.45 * (0.5 * cu + 0.5 * cl)
        out.setdefault(k[0], []).append({
            "camera_id": k[0], "track_id": k[1], "score": round(score, 3), "appearance": round(float(sims[i]), 3),
            "start": live[k][0], "end": live[k][1],
            "clothing": {"upper": ca.get("upper_color"), "lower": ca.get("lower_color")},
            "thumb_url": f"/web/thumb?camera_id={k[0]}&track_id={k[1]}"})
    cams = []
    for cam, rows in out.items():
        rows.sort(key=lambda r: -r["score"])
        cams.append({"camera_id": cam, "has_start_time": cam in offs, "candidates": rows[:per_camera]})
    cams.sort(key=lambda c: -c["candidates"][0]["score"])
    return {"seed": {"camera_id": camera_id, "track_id": int(track_id), "clothing": {"upper": sa.get("upper_color"), "lower": sa.get("lower_color")},
                     "thumb_url": f"/web/thumb?camera_id={camera_id}&track_id={int(track_id)}"},
            "cameras": cams,
            "missing_start_times": sorted({c["camera_id"] for c in cams if not c["has_start_time"]} | ({camera_id} if camera_id not in offs else set()))}
