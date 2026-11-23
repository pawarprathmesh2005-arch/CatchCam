"""
Clothing attributes for tracked people: upper-body colour and lower-body colour, voted over many frames.

No models, no downloads: colour naming reuses vehicle_intel.core.classify_color (HSV pixel vote).
Used for three things:
  1. search      "person in a red jacket and blue jeans"  -> ranked tracks   (parse_description + score_track)
  2. verifying   a cross-camera Re-ID match whose clothing clearly differs is flagged   (match_agreement)
  3. Q&A         "what was track 5 wearing?"

Deliberately NOT here: face recognition. Clothing is weaker evidence than a face, and much easier to defend
privacy-wise; it is also not stable over days, so treat it as same-day evidence only.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from vehicle_intel.core import classify_color

# ----------------------------------------------------------------------------- vocabulary
UPPER_WORDS = {"jacket", "shirt", "tshirt", "t-shirt", "tee", "top", "hoodie", "sweater", "coat", "kurta",
               "blouse", "jumper", "vest", "blazer", "sweatshirt", "uniform"}
LOWER_WORDS = {"pants", "pant", "trousers", "jeans", "shorts", "skirt", "leggings", "bottoms", "slacks"}
COLOR_ALIASES = {"grey": "gray", "navy": "blue", "maroon": "red", "crimson": "red", "violet": "purple",
                 "tan": "brown", "beige": "silver", "cream": "white", "khaki": "brown", "golden": "yellow"}
COLORS = {"black", "white", "gray", "silver", "red", "orange", "yellow", "green", "blue", "purple", "brown"}
# colours the HSV vote can plausibly confuse with each other under CCTV lighting -> partial credit
NEIGHBOURS = {frozenset(p) for p in [("white", "silver"), ("silver", "gray"), ("gray", "black"),
                                      ("orange", "brown"), ("orange", "yellow"), ("red", "brown"), ("red", "orange")]}


def body_regions(crop_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Upper-body (torso) and lower-body (legs) strips of a standing-person crop; head and feet excluded."""
    h, w = crop_bgr.shape[:2]
    if h < 48 or w < 16:
        return None, None
    upper = crop_bgr[int(h * 0.18):int(h * 0.50), int(w * 0.15):int(w * 0.85)]
    lower = crop_bgr[int(h * 0.55):int(h * 0.92), int(w * 0.15):int(w * 0.85)]
    return (upper if upper.size else None), (lower if lower.size else None)


def scene_levels(frames_bgr) -> Tuple[int, int]:
    """(black_level, white_level) of a camera, from frames spread over its footage.

    On a washed-out or over-exposed camera a black T-shirt measures ~100 on the 0-255 brightness scale instead of ~30, so a fixed
    'black is below 55' rule calls it gray. The scene's own darkest and brightest areas tell us where black and white really are
    (the median of many frames is mostly the empty background, without the people)."""
    import cv2
    small = [cv2.resize(f, (160, 90)) for f in frames_bgr if f is not None]
    if not small:
        return 0, 255
    v = cv2.cvtColor(np.median(np.stack(small), axis=0).astype(np.uint8), cv2.COLOR_BGR2HSV)[..., 2].ravel()
    black = int(min(np.percentile(v, 3), 90))          # never trust a 'black level' above 90 (a scene with nothing dark in it)
    white = int(max(np.percentile(v, 97), 150))
    return black, white


def classify_clothing(region_bgr: np.ndarray, levels: Optional[Tuple[int, int]] = None) -> Tuple[str, float]:
    """Dominant colour of a body strip -> (name, share). Auto-levels with the camera's black/white levels, names pale
    colours too (clothing is often washed out), and ignores bright floor/wall pixels around the person when there are enough others."""
    import cv2
    roi = cv2.resize(region_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    black, white = levels or (0, 255)
    span = max(white - black, 80)
    gain = 255.0 / span
    H = hsv[:, 0] * 2
    S = np.clip(hsv[:, 1] * gain, 0, 255)
    V = np.clip((hsv[:, 2] - black) * gain, 0, 255)
    floor = (S < 40) & (V >= 215)                       # the white wall/floor next to the body
    keep = ~floor
    if keep.mean() >= 0.25:
        H, S, V = H[keep], S[keep], V[keep]
    names = np.full(len(V), "gray", dtype=object)
    low = S < 38
    names[low & (V >= 190)] = "white"
    names[low & (V >= 120) & (V < 190)] = "silver"
    names[V < 55] = "black"
    chroma = (S >= 38) & (V >= 55)
    for name, m in [("red", (H < 12) | (H >= 345)), ("orange", (H >= 12) & (H < 40)), ("yellow", (H >= 40) & (H < 70)),
                    ("green", (H >= 70) & (H < 165)), ("blue", (H >= 165) & (H < 255)), ("purple", (H >= 255) & (H < 345))]:
        names[chroma & m] = name
    names[chroma & (H >= 12) & (H < 40) & (V < 140)] = "brown"
    name, n = Counter(names.tolist()).most_common(1)[0]
    return name, n / len(names)


def analyze_person(crop_bgr: np.ndarray, levels: Optional[Tuple[int, int]] = None) -> dict:
    """One frame -> {'upper': (colour, share), 'lower': (colour, share)}; (None, 0) when the crop is too small.
    levels = the camera's (black, white) brightness, from scene_levels(); without it the plain rules are used."""
    upper, lower = body_regions(crop_bgr)
    out = {}
    for name, region in (("upper", upper), ("lower", lower)):
        if region is None:
            out[name] = (None, 0.0)
        elif levels is None:
            out[name] = classify_color(region, central=False)
        else:
            out[name] = classify_clothing(region, levels)
    return out


@dataclass
class PersonAggregator:
    """Votes per-frame colours into one answer per track (weighted by how dominant the colour was in each frame)."""
    camera_id: str
    track_id: int
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    n: int = 0
    _votes: Dict[str, Dict[str, float]] = field(default_factory=lambda: {"upper": defaultdict(float), "lower": defaultdict(float)})

    def add(self, ts: float, result: dict) -> None:
        self.first_ts = ts if self.first_ts is None else min(self.first_ts, ts)
        self.last_ts = ts if self.last_ts is None else max(self.last_ts, ts)
        self.n += 1
        for part in ("upper", "lower"):
            colour, share = result.get(part, (None, 0.0))
            if colour and colour != "unknown":
                self._votes[part][colour] += max(share, 0.05)

    def result(self) -> dict:
        row = {"camera_id": self.camera_id, "track_id": self.track_id, "num_samples": self.n,
               "first_seen_timestamp": self.first_ts, "last_seen_timestamp": self.last_ts}
        for part in ("upper", "lower"):
            v = self._votes[part]
            total = sum(v.values())
            best = max(v, key=v.get) if v else None
            row[f"{part}_color"] = best
            row[f"{part}_conf"] = round(v[best] / total, 2) if best else 0.0
        return row


# ----------------------------------------------------------------------------- matching colours
def color_similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return 0.5 if frozenset((a, b)) in NEIGHBOURS else 0.0


# ----------------------------------------------------------------------------- text queries
def _norm_color(tok: str) -> Optional[str]:
    tok = COLOR_ALIASES.get(tok, tok)
    return tok if tok in COLORS else None


def normalize_color(word: Optional[str]) -> Optional[str]:
    """'Grey' -> 'gray', 'navy' -> 'blue'; None if it is not a colour we know."""
    return _norm_color((word or "").strip().lower())


def parse_description(text: str) -> Dict[str, Optional[str]]:
    """
    'person wearing a red jacket and blue jeans' -> {'upper': 'red', 'lower': 'blue', 'any': None}
    A colour with no garment ('someone in green') goes to 'any' (matches upper OR lower).
    Understands 'red jacket', 'jacket in red', 'shirt is blue', 'dark blue shirt'.
    """
    toks = re.findall(r"[a-z\-]+", (text or "").lower())
    want: Dict[str, Optional[str]] = {"upper": None, "lower": None, "any": None}
    used = set()
    for i, tok in enumerate(toks):
        slot = "upper" if tok in UPPER_WORDS else "lower" if tok in LOWER_WORDS else None
        if not slot:
            continue
        cand = None
        for j in (i - 1, i - 2):                                    # colour just before: 'red jacket', 'dark blue shirt'
            if 0 <= j < len(toks) and j not in used and _norm_color(toks[j]):
                cand = j; break
        if cand is None:
            for j in (i + 1, i + 2, i + 3):                         # 'jacket in red', 'shirt is blue'
                if j < len(toks) and j not in used and _norm_color(toks[j]):
                    cand = j; break
                if j < len(toks) and toks[j] in UPPER_WORDS | LOWER_WORDS:
                    break
        if cand is not None:
            want[slot] = _norm_color(toks[cand]); used.add(cand)
    leftovers = [_norm_color(t) for k, t in enumerate(toks) if k not in used and _norm_color(t)]
    if leftovers and not (want["upper"] or want["lower"]):
        want["any"] = leftovers[0]
    return want


def score_track(row: dict, want: Dict[str, Optional[str]]) -> float:
    """0..1: how well a track's clothing matches the description; confidence-weighted. 0 = no match."""
    parts = []
    for slot in ("upper", "lower"):
        if want.get(slot):
            parts.append(color_similarity(row.get(f"{slot}_color"), want[slot]) * max(row.get(f"{slot}_conf", 0), 0.3))
    if want.get("any"):
        parts.append(max(color_similarity(row.get(f"{s}_color"), want["any"]) * max(row.get(f"{s}_conf", 0), 0.3)
                         for s in ("upper", "lower")))
    return round(sum(parts) / len(parts), 3) if parts else 0.0


def matches_all(row: dict, want: Dict[str, Optional[str]]) -> bool:
    """True only if EVERY colour mentioned in the description is at least a near match for this person."""
    for slot in ("upper", "lower"):
        if want.get(slot) and color_similarity(row.get(f"{slot}_color"), want[slot]) == 0.0:
            return False
    if want.get("any") and max(color_similarity(row.get(f"{s}_color"), want["any"]) for s in ("upper", "lower")) == 0.0:
        return False
    return any(want.values())


def search_people(rows: List[dict], description: str, camera_id: Optional[str] = None, limit: int = 10) -> List[dict]:
    want = parse_description(description)
    if not any(want.values()):
        return []
    hits = []
    for r in rows:
        if camera_id and r["camera_id"] != camera_id:
            continue
        s = score_track(r, want)
        if s > 0:
            hits.append({**r, "score": s})
    hits.sort(key=lambda r: (-r["score"], r.get("first_seen_timestamp") or 0))
    return hits[:limit]


# ----------------------------------------------------------------------------- verifying Re-ID matches
def match_agreement(a: Optional[dict], b: Optional[dict], min_conf: float = 0.5) -> dict:
    """
    Do two tracks' clothing agree? verdict: 'consistent' | 'conflict' | 'unknown'.
    'conflict' only when BOTH upper and lower are confidently different (one changed layer, a jacket taken off,
    or a lighting shift should not trigger it).
    """
    if not a or not b:
        return {"verdict": "unknown", "upper": None, "lower": None, "reason": "no clothing data for one of the tracks"}
    sims, confident = {}, {}
    for slot in ("upper", "lower"):
        ca, cb = a.get(f"{slot}_color"), b.get(f"{slot}_color")
        sims[slot] = color_similarity(ca, cb) if ca and cb else None
        confident[slot] = bool(ca and cb and a.get(f"{slot}_conf", 0) >= min_conf and b.get(f"{slot}_conf", 0) >= min_conf)
    known = [s for s in ("upper", "lower") if sims[s] is not None]
    if not known:
        return {"verdict": "unknown", "upper": sims["upper"], "lower": sims["lower"], "reason": "clothing colours missing"}
    if all(sims[s] == 0.0 and confident[s] for s in known) and len(known) == 2:
        return {"verdict": "conflict", "upper": sims["upper"], "lower": sims["lower"],
                "reason": f"upper {a['upper_color']} vs {b['upper_color']}, lower {a['lower_color']} vs {b['lower_color']}"}
    if any(sims[s] and sims[s] >= 0.5 for s in known):
        return {"verdict": "consistent", "upper": sims["upper"], "lower": sims["lower"], "reason": "clothing colours agree"}
    return {"verdict": "unknown", "upper": sims["upper"], "lower": sims["lower"],
            "reason": "colours differ but not confidently enough to call a conflict"}
