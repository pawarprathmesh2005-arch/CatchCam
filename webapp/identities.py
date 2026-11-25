"""One 'Person' number per real person, across all cameras and across every time they come back.

The tracker only numbers a person *within one camera and one continuous appearance*: walk out of view and back, or onto the next
camera, and you get a new track id. This groups those tracks into people (P1, P2, ...):
  * tracks are compared the same way as in follow.py (appearance with the average removed + clothing colours),
  * two tracks that are visible on the SAME camera at the SAME time can never be the same person,
  * when both cameras have start times, tracks far apart in time are not linked,
  * tracks are merged best-pair-first (average linkage) until no pair of groups is similar enough.
It is a best guess made from appearance, so it can be wrong; the threshold is adjustable.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from webapp.follow import _load, _read, colour_score

THRESHOLD = 0.62          # average pair score needed to join two groups (0..1)
MAX_TIME_GAP = 900.0      # seconds; only used when both cameras have start times
MIN_FRAMES_NOTE = 0


def _scores(keys, emb, attrs):
    X = np.stack([emb[k] for k in keys])
    X = X - X.mean(0)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(n == 0, 1, n)
    cos = X @ X.T
    app = np.clip((cos + 0.1) / 0.6, 0, 1)
    m = len(keys)
    col = np.zeros((m, m))
    up = [attrs.get(k, {}).get("upper_color") for k in keys]
    lo = [attrs.get(k, {}).get("lower_color") for k in keys]
    for i in range(m):
        for j in range(i + 1, m):
            col[i, j] = col[j, i] = 0.5 * colour_score(up[i], up[j]) + 0.5 * colour_score(lo[i], lo[j])
    return 0.55 * app + 0.45 * col


def _ends(d):
    """For every track: (time, box) of its first and last sighting, so a track that ends where and when another begins can be joined."""
    ends = {}
    for f in sorted((d / "outputs").glob("*_observations.json")):
        if ".before_cleanup" in f.name:
            continue
        cam = f.name[: -len("_observations.json")]
        for r in _read(f, []):
            if isinstance(r, dict) and r.get("class", "person") == "person" and r.get("bbox_xyxy"):
                k = (cam, int(r["track_id"]))
                a, b = ends.get(k, (None, None))
                t, box = r["timestamp"], r["bbox_xyxy"]
                if a is None or t < a[0]:
                    a = (t, box)
                if b is None or t > b[0]:
                    b = (t, box)
                ends[k] = (a, b)
    return ends


def _handover(prev, nxt, max_gap=4.0, reach=1.3):
    """True when `prev` stops and `nxt` starts on the same camera a moment later, in about the same place (the tracker lost the person
    for a second, e.g. behind a pillar or another person, and gave them a new number)."""
    (t1, b1), (t2, b2) = prev[1], nxt[0]
    gap = t2 - t1
    if gap < 0 or gap > max_gap:
        return False
    h = max(b1[3] - b1[1], b2[3] - b2[1], 1.0)
    cx1, cy1, cx2, cy2 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2, (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 <= reach * h


def build(data_dir, threshold=THRESHOLD, max_time_gap=MAX_TIME_GAP):
    d = Path(data_dir)
    live, attrs, emb = _load(d)
    offs = _read(d / "time_offsets.json", {})
    keys = sorted(k for k in emb if k in live)
    if len(keys) < 2:
        groups = [[k] for k in keys]
    else:
        S = _scores(keys, emb, attrs)
        m = len(keys)
        ends = _ends(d)
        blocked = np.zeros((m, m), bool)
        for i in range(m):
            for j in range(i + 1, m):
                (ca, ta), (cb, tb) = keys[i], keys[j]
                a0, a1 = live[keys[i]]
                b0, b1 = live[keys[j]]
                if ca == cb and a0 <= b1 and b0 <= a1:
                    blocked[i, j] = blocked[j, i] = True                       # together on one camera: two different people
                elif ca != cb and ca in offs and cb in offs and max(b0 - a1, a0 - b1, 0.0) > max_time_gap:
                    blocked[i, j] = blocked[j, i] = True
                elif ca == cb and keys[i] in ends and keys[j] in ends:
                    first, second = (keys[i], keys[j]) if a0 <= b0 else (keys[j], keys[i])
                    if _handover(ends[first], ends[second]):
                        S[i, j] = S[j, i] = max(S[i, j], 0.95)          # same camera, same place, a moment later: the same person
        # average linkage, kept as running sums so each merge costs one matrix row, not a rescan of every pair
        sums = S.copy()
        np.fill_diagonal(sums, 0)
        size = np.ones(m)
        block = blocked.astype(np.int32)
        alive = np.ones(m, bool)
        members = {i: [i] for i in range(m)}
        while True:
            avg = sums / np.outer(size, size)
            avg[~alive, :] = -1
            avg[:, ~alive] = -1
            avg[block > 0] = -1
            np.fill_diagonal(avg, -1)
            i, j = np.unravel_index(np.argmax(avg), avg.shape)
            if avg[i, j] <= threshold:
                break
            sums[i, :] += sums[j, :]
            sums[:, i] += sums[:, j]
            block[i, :] += block[j, :]
            block[:, i] += block[:, j]
            size[i] += size[j]
            members[i] += members.pop(j)
            alive[j] = False
            sums[i, i] = 0
            block[i, i] = 0
        groups = [members[i] for i in sorted(members)]
        groups = [[keys[i] for i in g] for g in groups]
    people = []
    for g in groups:
        members = sorted(({"camera_id": c, "track_id": t, "start": live[(c, t)][0], "end": live[(c, t)][1]} for c, t in g), key=lambda r: r["start"])
        up = Counter(attrs.get(k, {}).get("upper_color") for k in g if attrs.get(k, {}).get("upper_color"))
        lo = Counter(attrs.get(k, {}).get("lower_color") for k in g if attrs.get(k, {}).get("lower_color"))
        people.append({"members": members, "cameras": sorted({r["camera_id"] for r in members}),
                       "appearances": len(members), "start": members[0]["start"], "end": members[-1]["end"],
                       "clothing": {"upper": up.most_common(1)[0][0] if up else None, "lower": lo.most_common(1)[0][0] if lo else None}})
    people.sort(key=lambda p: (-len(p["cameras"]), -p["appearances"], p["start"]))
    for n, p in enumerate(people, 1):
        p["person_id"] = f"P{n}"
    return {"threshold": threshold, "tracks": len(keys), "people": people}


def of_track(result, camera_id, track_id):
    for p in result["people"]:
        if any(r["camera_id"] == camera_id and r["track_id"] == int(track_id) for r in p["members"]):
            return p
    return None


_cache = {}


def get(data_dir, threshold=THRESHOLD):
    """Cached until a results file changes."""
    d = Path(data_dir)
    stamp = tuple(sorted((str(p), p.stat().st_mtime_ns) for pat in ("outputs/*_observations.json", "outputs/*_attributes.json", "embeddings/*_embeddings.json", "time_offsets.json")
                         for p in d.glob(pat)))
    key = (str(d.resolve()), threshold)
    if key not in _cache or _cache[key][0] != stamp:
        _cache[key] = (stamp, build(d, threshold))
    return _cache[key][1]
