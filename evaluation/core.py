"""
CatchCam — accuracy evaluation.

You say, by watching the footage, which tracks are the same person ("ground truth"). This module then scores what CatchCam
decided on its own and lists exactly where it went wrong.

Ground truth is stored in <data_dir>/ground_truth.json as  {"labels": [{"camera_id": "cam1", "track_id": 3, "person": "P1"}, ...]}.
Only labelled tracks are scored, so you can label a part of the footage.  Unlabelled tracks are ignored.

What is measured (all cross-camera, because that is what the matching step does):
  * pairwise precision / recall / F1   every pair of labelled tracks on different cameras is either "same person" or not;
                                       CatchCam said "same journey" or not.
  * fully correct journeys             people whose whole route was rebuilt exactly (no missing and no extra track)
  * Rank-1 / Rank-5 / mAP              (needs embeddings/*_embeddings.json) for each track, rank the tracks on the other cameras
                                       by appearance similarity: is a true match first?  Standard re-identification numbers.
  * threshold sweep                    (needs embeddings) precision / recall if you used a different similarity cut-off
  * fragmentation                      people with more than one track on the SAME camera (the tracker lost and re-found them)

Not measured: frame-by-frame tracking accuracy (IDF1/MOTA). That needs a bounding box labelled on every frame, which is a
separate and much bigger labelling job. Do not quote these numbers as IDF1.
"""
from __future__ import annotations

import json
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from build_graph import node_id
from missing_camera import get_full_journey
from qa_engine.engine import Store

GT_FILE = "ground_truth.json"
RESULT_FILE = "evaluation_result.json"
Key = Tuple[str, int]
_PERSON = re.compile(r"^[\w .\-]{1,40}$")


# ----------------------------------------------------------------------------- ground truth on disk
def load_labels(data_dir) -> Dict[Key, str]:
    f = Path(data_dir) / GT_FILE
    if not f.exists():
        return {}
    rows = json.loads(f.read_text(encoding="utf-8-sig")).get("labels", [])
    return {(r["camera_id"], int(r["track_id"])): str(r["person"]) for r in rows}


def save_labels(data_dir, rows: List[dict]) -> int:
    """rows: [{camera_id, track_id, person}]. Rows with an empty person are dropped (= unlabelled). Returns how many were kept."""
    known = {k for k in all_track_keys(data_dir)}
    out: Dict[Key, str] = {}
    for r in rows or []:
        person = str(r.get("person") or "").strip()
        if not person:
            continue
        if not _PERSON.match(person):
            raise ValueError(f"'{person}' is not a usable person name. Use letters, numbers, spaces, dots, dashes (max 40).")
        try:
            key = (str(r["camera_id"]), int(r["track_id"]))
        except (KeyError, TypeError, ValueError):
            raise ValueError("Each label needs camera_id, track_id and person.")
        if key not in known:
            raise ValueError(f"{key[0]} track {key[1]} does not exist in the data.")
        out[key] = person
    payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "labels": [{"camera_id": c, "track_id": t, "person": p} for (c, t), p in sorted(out.items())]}
    f = Path(data_dir) / GT_FILE
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(f)
    return len(out)


# ----------------------------------------------------------------------------- what exists
def _track_table(data_dir) -> Dict[Key, dict]:
    store = Store(data_dir)
    tracks: Dict[Key, dict] = {}
    for f in store.obs_files():
        if f.name.endswith("_observations_live.json"):
            continue
        for r in json.loads(f.read_text(encoding="utf-8-sig")):
            k = (r["camera_id"], int(r["track_id"]))
            t = r["timestamp"]
            e = tracks.setdefault(k, {"camera_id": k[0], "track_id": k[1], "first_seen": t, "last_seen": t})
            e["first_seen"], e["last_seen"] = min(e["first_seen"], t), max(e["last_seen"], t)
    return tracks


def all_track_keys(data_dir) -> List[Key]:
    return sorted(_track_table(data_dir))


def _predicted_chains(data_dir) -> Dict[Key, frozenset]:
    """For every track: the set of tracks CatchCam puts on the same journey (including itself)."""
    G = Store(data_dir).graph()
    chains: Dict[Key, frozenset] = {}
    if G is None:
        return chains
    for n, d in G.nodes(data=True):
        j = get_full_journey(G, d["camera_id"], d["track_id"])
        chains[(d["camera_id"], d["track_id"])] = frozenset((h["camera_id"], h["track_id"]) for h in j)
    return chains


def track_list(data_dir) -> dict:
    """Everything the labelling screen needs: each track, its label, and which journey CatchCam put it on."""
    tracks, labels, chains = _track_table(data_dir), load_labels(data_dir), _predicted_chains(data_dir)
    group_ids: Dict[frozenset, int] = {}
    rows = []
    for k in sorted(tracks, key=lambda k: (tracks[k]["first_seen"], k)):
        ch = chains.get(k, frozenset([k]))
        gid = group_ids.setdefault(ch, len(group_ids) + 1)
        rows.append({**tracks[k], "label": labels.get(k, ""), "predicted_group": gid,
                     "thumb_url": f"/web/thumb?camera_id={k[0]}&track_id={k[1]}"})
    return {"tracks": rows, "labelled": sum(1 for r in rows if r["label"]), "people": sorted(set(labels.values()))}


# ----------------------------------------------------------------------------- embeddings (optional)
def load_embeddings(data_dir) -> Dict[Key, np.ndarray]:
    out: Dict[Key, np.ndarray] = {}
    folder = Path(data_dir) / "embeddings"
    for f in sorted(folder.glob("*_embeddings.json")) if folder.exists() else []:
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        for tid, v in d.get("tracks", {}).items():
            vec = np.asarray(v["embedding"], dtype=float)
            n = np.linalg.norm(vec)
            if n > 0:
                out[(d["camera_id"], int(tid))] = vec / n
    return out


def _average_precision(relevant: List[bool]) -> float:
    hits, total = 0, 0.0
    for i, r in enumerate(relevant, 1):
        if r:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def ranking_metrics(labels: Dict[Key, str], emb: Dict[Key, np.ndarray]) -> Optional[dict]:
    keys = [k for k in labels if k in emb]
    if len(keys) < 2:
        return None
    r1 = r5 = 0
    aps: List[float] = []
    for q in keys:
        gallery = [g for g in keys if g[0] != q[0]]
        rel_total = [g for g in gallery if labels[g] == labels[q]]
        if not rel_total:
            continue                                             # nobody to find on another camera: not a valid query
        gallery.sort(key=lambda g: -float(emb[q] @ emb[g]))
        flags = [labels[g] == labels[q] for g in gallery]
        r1 += flags[0]
        r5 += any(flags[:5])
        aps.append(_average_precision(flags))
    if not aps:
        return None
    n = len(aps)
    return {"queries": n, "rank1": r1 / n, "rank5": r5 / n, "map": float(np.mean(aps))}


def threshold_sweep(labels: Dict[Key, str], emb: Dict[Key, np.ndarray], steps=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)) -> List[dict]:
    pairs = [(a, b) for a, b in combinations([k for k in labels if k in emb], 2) if a[0] != b[0]]
    if not pairs:
        return []
    sims = np.array([float(emb[a] @ emb[b]) for a, b in pairs])
    truth = np.array([labels[a] == labels[b] for a, b in pairs])
    out = []
    for th in steps:
        pred = sims >= th
        tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        out.append({"threshold": th, "precision": p, "recall": r, "f1": _f1(p, r), "false_matches": fp, "missed": fn})
    return out


def _f1(p, r):
    return None if p is None or r is None or p + r == 0 else 2 * p * r / (p + r)


# ----------------------------------------------------------------------------- the evaluation
def evaluate(data_dir) -> dict:
    d = Path(data_dir)
    labels = load_labels(d)
    tracks = _track_table(d)
    labels = {k: v for k, v in labels.items() if k in tracks}          # labels for tracks that no longer exist are ignored
    if len(labels) < 2 or len(set(labels.values())) < 1:
        return {"status": "need_labels", "message": "Label at least two tracks first (and at least one person who appears on two cameras).",
                "labelled": len(labels)}
    chains = _predicted_chains(d)
    keys = sorted(labels)

    def predicted_same(a: Key, b: Key) -> bool:
        return b in chains.get(a, frozenset()) or a in chains.get(b, frozenset())

    tp = fp = fn = tn = 0
    false_matches, missed = [], []
    for a, b in combinations(keys, 2):
        if a[0] == b[0]:
            continue                                                    # same-camera pairs are the tracker's job, see fragmentation
        truth, pred = labels[a] == labels[b], predicted_same(a, b)
        if truth and pred:
            tp += 1
        elif pred:
            fp += 1
            false_matches.append({"a": f"{a[0]} #{a[1]}", "b": f"{b[0]} #{b[1]}", "a_person": labels[a], "b_person": labels[b]})
        elif truth:
            fn += 1
            missed.append({"a": f"{a[0]} #{a[1]}", "b": f"{b[0]} #{b[1]}", "person": labels[a]})
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None

    by_person: Dict[str, List[Key]] = {}
    for k, p in labels.items():
        by_person.setdefault(p, []).append(k)
    # CatchCam's own grouping of the labelled tracks: tracks it linked, directly or through others, form one predicted journey
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in combinations(keys, 2):
        if predicted_same(a, b):
            parent[find(a)] = find(b)
    components: Dict[Key, set] = {}
    for k in keys:
        components.setdefault(find(k), set()).add(k)
    journeys, correct = [], 0
    for person, ks in sorted(by_person.items()):
        if len({k[0] for k in ks}) < 2:
            continue                                                    # a person seen on one camera has no journey to rebuild
        groups = {find(k) for k in ks}
        merged = set().union(*(components[g] for g in groups))
        ok = len(groups) == 1 and merged == set(ks)                     # one predicted journey, with exactly this person's tracks
        correct += ok
        journeys.append({"person": person, "tracks": [f"{c} #{t}" for c, t in sorted(ks)], "correct": ok,
                         "extra": [f"{c} #{t} ({labels[(c, t)]})" for c, t in sorted(merged - set(ks))],
                         "missing": [f"{c} #{t}" for c, t in sorted(set(ks) - components[find(sorted(ks)[0])])]})
    fragmented = [{"person": p, "camera_id": c, "tracks": sorted(t for cc, t in ks if cc == c)}
                  for p, ks in sorted(by_person.items()) for c in sorted({k[0] for k in ks}) if sum(1 for k in ks if k[0] == c) > 1]

    emb = load_embeddings(d)
    ranking = ranking_metrics(labels, emb)
    result = {
        "status": "ok", "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "labelled_tracks": len(labels), "people": len(by_person), "cameras": sorted({k[0] for k in labels}),
        "pairs": {"checked": tp + fp + fn + tn, "true_same": tp + fn, "correct_matches": tp, "false_matches": fp, "missed_matches": fn},
        "precision": precision, "recall": recall, "f1": _f1(precision, recall),
        "journeys": {"with_journey": len(journeys), "fully_correct": correct, "details": journeys},
        "ranking": ranking, "threshold_sweep": threshold_sweep(labels, emb) if emb else [],
        "fragmentation": fragmented, "false_match_list": false_matches[:50], "missed_match_list": missed[:50],
        "notes": [],
    }
    if ranking is None:
        result["notes"].append("Rank-1 / mAP were skipped: they need embeddings/*_embeddings.json and at least one person labelled on two cameras.")
    if tp + fn == 0:
        result["notes"].append("No person is labelled on two different cameras yet, so recall cannot be measured. Label the people who walk between cameras.")
    if len(labels) < 20:
        result["notes"].append(f"Only {len(labels)} tracks are labelled. With so few, these percentages can swing a lot; say how many tracks you used when you quote them.")
    (d / RESULT_FILE).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


# ----------------------------------------------------------------------------- a write-up you can paste into the report
def _pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def to_markdown(r: dict) -> str:
    if r.get("status") != "ok":
        return "# CatchCam accuracy evaluation\n\n" + r.get("message", "No result yet.") + "\n"
    p = r["pairs"]
    lines = ["# CatchCam accuracy evaluation", "",
             f"Evaluated {r['evaluated_at']} on {r['labelled_tracks']} hand-labelled tracks of {r['people']} people across {', '.join(r['cameras'])}.", "",
             "## Cross-camera matching", "", "| Measure | Value |", "|---|---|",
             f"| Precision | {_pct(r['precision'])} |", f"| Recall | {_pct(r['recall'])} |", f"| F1 | {_pct(r['f1'])} |",
             f"| Correct / false / missed matches | {p['correct_matches']} / {p['false_matches']} / {p['missed_matches']} |",
             f"| Journeys rebuilt exactly | {r['journeys']['fully_correct']} of {r['journeys']['with_journey']} |"]
    if r.get("ranking"):
        k = r["ranking"]
        lines += ["", "## Re-identification ranking", "", "| Measure | Value |", "|---|---|", f"| Queries | {k['queries']} |",
                  f"| Rank-1 | {_pct(k['rank1'])} |", f"| Rank-5 | {_pct(k['rank5'])} |", f"| mAP | {_pct(k['map'])} |"]
    if r.get("threshold_sweep"):
        lines += ["", "## Similarity threshold sweep (raw embedding similarity)", "", "| Threshold | Precision | Recall | F1 | False | Missed |", "|---|---|---|---|---|---|"]
        lines += [f"| {s['threshold']:.2f} | {_pct(s['precision'])} | {_pct(s['recall'])} | {_pct(s['f1'])} | {s['false_matches']} | {s['missed']} |" for s in r["threshold_sweep"]]
    if r["fragmentation"]:
        lines += ["", "## Track fragmentation", ""] + [f"- {f['person']} has {len(f['tracks'])} separate tracks on {f['camera_id']}: {f['tracks']}" for f in r["fragmentation"]]
    if r["false_match_list"]:
        lines += ["", "## False matches (different people linked)", ""] + [f"- {m['a']} ({m['a_person']}) linked with {m['b']} ({m['b_person']})" for m in r["false_match_list"]]
    if r["missed_match_list"]:
        lines += ["", "## Missed matches (same person not linked)", ""] + [f"- {m['person']}: {m['a']} and {m['b']}" for m in r["missed_match_list"]]
    if r["notes"]:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in r["notes"]]
    lines += ["", "Frame-by-frame tracking accuracy (IDF1/MOTA) is not measured here: it needs a box labelled on every frame.", ""]
    return "\n".join(lines)
