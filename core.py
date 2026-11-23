"""
Vehicle intelligence: type, colour and number plate from vehicle crops, plus per-track voting.

Pure NumPy + OpenCV. OCR backend is pluggable: EasyOCR if installed, else pytesseract.
Plate rules are for Indian plates (e.g. MH12AB1234); see normalize_plate() to adapt.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# COCO class ids used by the standard YOLO weights
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ----------------------------------------------------------------------------- colour

def classify_color(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Dominant body colour of a vehicle crop -> (name, share 0..1).
    Uses the central region (less road/background), HSV rules, and a vote over pixels.
    Names: black white gray silver red orange yellow green blue purple brown
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0
    h, w = crop_bgr.shape[:2]
    roi = crop_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.15):int(w * 0.85)]
    if roi.size == 0:
        roi = crop_bgr
    roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    H, S, V = hsv[:, 0] * 2, hsv[:, 1], hsv[:, 2]          # H in degrees 0..358

    names = np.full(len(hsv), "gray", dtype=object)
    names[V < 55] = "black"
    chroma = (S >= 70) & (V >= 55)
    names[(S < 40) & (V >= 190)] = "white"
    names[(S < 40) & (V >= 120) & (V < 190)] = "silver"
    names[(S < 40) & (V >= 55) & (V < 120)] = "gray"
    # low-ish saturation in-between: treat as gray/silver by brightness
    mid = (S >= 40) & (S < 70) & (V >= 55)
    names[mid & (V >= 150)] = "silver"
    names[mid & (V < 150)] = "gray"
    hue_rules = [("red", (H < 12) | (H >= 345)), ("orange", (H >= 12) & (H < 40)),
                 ("yellow", (H >= 40) & (H < 70)), ("green", (H >= 70) & (H < 165)),
                 ("blue", (H >= 165) & (H < 255)), ("purple", (H >= 255) & (H < 345))]
    for name, m in hue_rules:
        names[chroma & m] = name
    # dark orange/brown
    names[chroma & (H >= 12) & (H < 40) & (V < 140)] = "brown"
    c = Counter(names.tolist())
    name, n = c.most_common(1)[0]
    return name, n / len(names)


# ----------------------------------------------------------------------------- plates

INDIAN_STATE_CODES = set("""AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY
RJ SK TN TS TR UK UP WB""".split())
_PLATE_RE = re.compile(r"^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})$")
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A"}


def normalize_plate(raw: str) -> Tuple[str, bool]:
    """
    Clean OCR text into an Indian plate. Returns (text, is_valid).
    Fixes position-aware look-alike confusions (O/0, I/1, S/5, B/8 ...): the first two characters
    must be letters, the next two digits, the last four digits, the middle 1-3 letters.
    """
    t = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if len(t) < 8 or len(t) > 11:
        return t, False
    chars = list(t)
    n = len(chars)
    for i in (0, 1):
        chars[i] = _TO_LETTER.get(chars[i], chars[i])
    for i in (2, 3):
        chars[i] = _TO_DIGIT.get(chars[i], chars[i])
    for i in range(n - 4, n):
        chars[i] = _TO_DIGIT.get(chars[i], chars[i])
    for i in range(4, n - 4):
        chars[i] = _TO_LETTER.get(chars[i], chars[i])
    fixed = "".join(chars)
    m = _PLATE_RE.match(fixed)
    return fixed, bool(m and m.group(1) in INDIAN_STATE_CODES)


def plate_distance(a: str, b: str) -> int:
    """Edit distance (Levenshtein) between two plate strings."""
    a, b = a or "", b or ""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def find_plate_region(crop_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Classical plate localisation inside a vehicle crop -> (x, y, w, h) or None.
    Replace with a trained plate detector for better recall (see README)."""
    h, w = crop_bgr.shape[:2]
    if h < 40 or w < 60:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    edges = cv2.Canny(gray, 80, 200)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < 8 or cw < 30:
            continue
        ar, area = cw / ch, cw * ch / (w * h)
        if not (1.8 <= ar <= 6.5) or not (0.01 <= area <= 0.35):
            continue
        patch = gray[y:y + ch, x:x + cw]
        score = float(patch.std()) * (1.0 + (y + ch / 2) / h)   # textured, and lower on the vehicle
        if score > best_score:
            best, best_score = (x, y, cw, ch), score
    return best


class PlateReader:
    """OCR wrapper. backend: 'auto' | 'easyocr' | 'tesseract'."""

    def __init__(self, backend: str = "auto"):
        self.backend, self._reader = None, None
        if backend in ("auto", "easyocr"):
            try:
                import easyocr
                self._reader, self.backend = easyocr.Reader(["en"], gpu=False, verbose=False), "easyocr"
            except Exception:
                if backend == "easyocr":
                    raise
        if self.backend is None:
            import pytesseract  # noqa: F401
            self.backend = "tesseract"

    @staticmethod
    def _variants(img: np.ndarray) -> List[np.ndarray]:
        """Preprocessing variants for OCR. The dark plate frame/border is trimmed first: it is read as
        spurious characters otherwise (this fixed 'H12081236' -> 'MH12AB1234' in testing)."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        m = max(2, int(min(g.shape[:2]) * 0.09))
        if g.shape[0] > 4 * m and g.shape[1] > 4 * m:
            g = g[m:-m, m:-m]
        scale = max(1.0, 60.0 / max(g.shape[0], 1)) * 3
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if otsu.mean() < 127:                      # light-on-dark plate -> make it dark text on white
            otsu = 255 - otsu
        pad = lambda x: cv2.copyMakeBorder(x, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        return [pad(otsu), pad(cv2.createCLAHE(2.0, (8, 8)).apply(g))]

    def read(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """-> (raw_text, confidence 0..1). Note: Tesseract often reports 0 confidence for single-line
        reads; the track-level vote then relies on plate validity and repetition instead."""
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
        variants = self._variants(plate_img)
        if self.backend == "easyocr":
            cands = []
            for v in variants:
                res = self._reader.readtext(v, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                if res:
                    cands.append(("".join(r[1] for r in res), float(np.mean([r[2] for r in res]))))
            return max(cands, key=lambda c: (normalize_plate(c[0])[1], c[1])) if cands else ("", 0.0)
        import pytesseract
        cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        cands = []
        for v in variants:
            d = pytesseract.image_to_data(v, config=cfg, output_type=pytesseract.Output.DICT)
            words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if w.strip() and float(c) >= 0]
            if words:
                cands.append(("".join(w for w, _ in words), float(np.mean([c for _, c in words])) / 100.0))
        if not cands:
            return "", 0.0
        # prefer a read that parses as a valid plate; otherwise the most confident one
        return max(cands, key=lambda c: (normalize_plate(c[0])[1], c[1]))


def analyze_vehicle(crop_bgr: np.ndarray, vehicle_type: str = "unknown",
                    reader: Optional[PlateReader] = None) -> dict:
    """Colour + plate for one vehicle crop (type comes from the detector's class)."""
    color, color_share = classify_color(crop_bgr)
    out = {"type": vehicle_type, "color": color, "color_share": round(color_share, 3),
           "plate_raw": "", "plate": "", "plate_valid": False, "plate_conf": 0.0}
    if reader is not None:
        box = find_plate_region(crop_bgr)
        if box:
            x, y, w, h = box
            raw, conf = reader.read(crop_bgr[y:y + h, x:x + w])
            plate, valid = normalize_plate(raw)
            out.update(plate_raw=raw, plate=plate, plate_valid=valid, plate_conf=round(conf, 3))
    return out


# ----------------------------------------------------------------------------- per-track voting

@dataclass
class TrackAggregator:
    """Collects per-frame results for one vehicle track and settles on a single answer."""
    camera_id: str
    track_id: int
    t_in: Optional[float] = None
    t_out: Optional[float] = None
    frames: int = 0
    _types: Counter = field(default_factory=Counter)
    _colors: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _plates: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _plates_raw: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, t: float, result: dict) -> None:
        self.t_in = t if self.t_in is None else min(self.t_in, t)
        self.t_out = t if self.t_out is None else max(self.t_out, t)
        self.frames += 1
        if result.get("type") and result["type"] != "unknown":
            self._types[result["type"]] += 1
        if result.get("color") and result["color"] != "unknown":
            self._colors[result["color"]] += max(result.get("color_share", 0.0), 0.05)
        p, conf = result.get("plate", ""), max(result.get("plate_conf", 0.0), 0.05)
        if p:
            (self._plates if result.get("plate_valid") else self._plates_raw)[p] += conf

    def result(self) -> dict:
        plates = self._plates or self._plates_raw
        plate = max(plates, key=plates.get) if plates else ""
        total = sum(plates.values()) or 1.0
        return {"camera_id": self.camera_id, "track_id": self.track_id,
                "vehicle_type": self._types.most_common(1)[0][0] if self._types else "unknown",
                "color": max(self._colors, key=self._colors.get) if self._colors else "unknown",
                "plate": plate, "plate_valid": bool(self._plates), "plate_support": round(plates.get(plate, 0) / total, 2),
                "first_seen_timestamp": self.t_in, "last_seen_timestamp": self.t_out, "num_frames": self.frames}
