"""
2D floor-plan fusion: put every sighting on a map of the building.

How: each camera is calibrated once with >= 4 point pairs (a spot in the camera image  <->  the same spot on the floor
plan). OpenCV turns that into a homography, which maps a person's FEET (bottom-centre of their bounding box, the point that
actually touches the floor) from image pixels to floor coordinates. A person's path on one camera becomes a polyline on the
plan; hops between cameras are joined, and unexplained time gaps (missing_camera.py) are drawn as dashed connectors.

Cameras with no calibration still appear: their sightings are pinned to the camera's position (marked approximate).

floorplan.json (in your data folder):
{
  "image": "floorplan.png",                 # optional background picture of the plan (same aspect as width x height)
  "width": 20.0, "height": 12.0,            # plan size in any unit you like (metres recommended)
  "scale": 40,                              # pixels per unit in rendered pictures (default 40)
  "cameras": {
    "cam1": {"position": [10, 0.5],
             "calibration": {"image_points": [[220,100],[420,100],[620,340],[20,340]],
                             "floor_points": [[0,0],[20,0],[20,12],[0,12]]}},
    "cam2": {"position": [19.5, 6]}         # no calibration: pinned at its position
  }
}
Tip: choose calibration points that are spread out and on the FLOOR (tile corners, door frames at floor level), 6-8 is better than 4.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PALETTE = [(200, 120, 30), (60, 160, 60), (40, 120, 230), (170, 60, 160), (150, 150, 20), (40, 60, 200)]   # BGR
AMBER, RED = (0, 170, 255), (50, 50, 230)


# ----------------------------------------------------------------------------- config + calibration
def load_config(path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for key in ("width", "height", "cameras"):
        if key not in cfg:
            raise ValueError(f"floorplan config is missing '{key}'")
    return cfg


def compute_homography(image_points, floor_points) -> np.ndarray:
    src, dst = np.float32(image_points), np.float32(floor_points)
    if len(src) != len(dst) or len(src) < 4:
        raise ValueError("calibration needs at least 4 matching image/floor point pairs")
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 0.05 * max(np.ptp(dst[:, 0]), np.ptp(dst[:, 1]), 1e-6)) if len(src) > 4 \
        else cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError("could not compute a homography (are the points collinear?)")
    return H


def project_points(H: np.ndarray, pts) -> np.ndarray:
    p = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H)
    return p.reshape(-1, 2)


def camera_homography(cam_cfg: dict) -> Optional[np.ndarray]:
    cal = cam_cfg.get("calibration")
    return compute_homography(cal["image_points"], cal["floor_points"]) if cal else None


def calibration_report(cfg: dict) -> Dict[str, dict]:
    """Per camera: calibrated?, reprojection error (RMS, plan units), and a verdict. A big error = bad clicks."""
    tol = 0.03 * max(cfg["width"], cfg["height"])
    out = {}
    for cam, c in cfg["cameras"].items():
        info = {"position": c.get("position"), "calibrated": False, "rms_error": None, "ok": None, "tolerance": round(tol, 3)}
        try:
            H = camera_homography(c)
            if H is not None:
                err = project_points(H, c["calibration"]["image_points"]) - np.float32(c["calibration"]["floor_points"])
                rms = float(np.sqrt((err ** 2).sum(axis=1).mean()))
                info.update(calibrated=True, rms_error=round(rms, 4), ok=rms <= tol)
        except ValueError as e:
            info["problem"] = str(e)
        out[cam] = info
    return out


# ----------------------------------------------------------------------------- paths
def foot_point(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def _smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    if len(a) < k:
        return a
    pad = k // 2
    padded = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.median(padded[i:i + k], axis=0) for i in range(len(a))])


def track_floor_path(observations: List[dict], camera_id: str, track_id: int, H: np.ndarray, max_points: int = 60) -> List[dict]:
    obs = sorted((o for o in observations if o["camera_id"] == camera_id and o["track_id"] == track_id), key=lambda o: o["frame"])
    if not obs:
        return []
    pts = project_points(H, [foot_point(o["bbox_xyxy"]) for o in obs])
    pts = _smooth(pts)
    idx = np.unique(np.linspace(0, len(obs) - 1, min(max_points, len(obs))).astype(int))
    return [{"t": obs[i]["timestamp"], "x": round(float(pts[i, 0]), 3), "y": round(float(pts[i, 1]), 3)} for i in idx]


def journey_on_plan(journey: List[dict], gaps: List[dict], cfg: dict, observations: List[dict]) -> dict:
    """journey: hops from get_full_journey(); gaps: from find_coverage_gaps()."""
    hops = []
    for h in journey:
        cam = h["camera_id"]
        cc = cfg["cameras"].get(cam)
        hop = {"camera_id": cam, "track_id": h["track_id"], "first_seen_timestamp": h["first_seen_timestamp"],
               "last_seen_timestamp": h["last_seen_timestamp"], "mode": "unmapped", "points": [], "out_of_bounds_ratio": 0.0}
        if cc is not None:
            H = camera_homography(cc)
            if H is not None:
                hop["points"] = track_floor_path(observations, cam, h["track_id"], H)
                hop["mode"] = "projected"
                if hop["points"]:
                    oob = sum(1 for p in hop["points"] if not (0 <= p["x"] <= cfg["width"] and 0 <= p["y"] <= cfg["height"]))
                    hop["out_of_bounds_ratio"] = round(oob / len(hop["points"]), 2)
            if not hop["points"] and cc.get("position"):
                x, y = cc["position"]
                hop["points"] = [{"t": h["first_seen_timestamp"], "x": x, "y": y}]
                hop["mode"] = "camera_position"
        hops.append(hop)
    by_pair = {(g["from_camera"], g["to_camera"], g["last_seen"]): g for g in gaps}
    connectors = []
    for a, b in zip(hops, hops[1:]):
        if not a["points"] or not b["points"]:
            continue
        g = by_pair.get((a["camera_id"], b["camera_id"], a["last_seen_timestamp"]))
        connectors.append({"from_camera": a["camera_id"], "to_camera": b["camera_id"],
                           "x1": a["points"][-1]["x"], "y1": a["points"][-1]["y"], "x2": b["points"][0]["x"], "y2": b["points"][0]["y"],
                           "gap_seconds": g["gap_seconds"] if g else round(b["first_seen_timestamp"] - a["last_seen_timestamp"], 2),
                           "unexplained": bool(g), "severity": (g or {}).get("severity")})
    return {"hops": hops, "connectors": connectors}


# ----------------------------------------------------------------------------- drawing
def _dashed(img, p1, p2, color, thick=2, dash=10):
    p1, p2 = np.float32(p1), np.float32(p2)
    L = float(np.linalg.norm(p2 - p1))
    if L < 1:
        return
    for i in range(0, int(L // dash) + 1, 2):
        a = p1 + (p2 - p1) * min(i * dash / L, 1)
        b = p1 + (p2 - p1) * min((i + 1) * dash / L, 1)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, thick, cv2.LINE_AA)


def _label(img, text, org, color, scale=0.5, thick=1):
    """Text with a white halo so it stays readable over lines; kept inside the picture."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = int(min(max(org[0], 4), img.shape[1] - tw - 4))
    y = int(min(max(org[1], th + 4), img.shape[0] - 4))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_png(cfg: dict, plan: Optional[dict] = None, data_dir=".", title: str = "") -> bytes:
    scale = float(cfg.get("scale", 40))
    W, Hh = int(cfg["width"] * scale), int(cfg["height"] * scale)
    img = np.full((Hh, W, 3), 246, np.uint8)
    bg = Path(data_dir) / cfg["image"] if cfg.get("image") else None
    if bg and bg.exists():
        pic = cv2.imread(str(bg))
        if pic is not None:
            img = cv2.addWeighted(cv2.resize(pic, (W, Hh)), 0.55, img, 0.45, 0)
    else:
        step = max(int(scale * (5 if cfg["width"] > 40 else 1)), 10)
        for x in range(0, W, step): cv2.line(img, (x, 0), (x, Hh), (232, 232, 232), 1)
        for y in range(0, Hh, step): cv2.line(img, (0, y), (W, y), (232, 232, 232), 1)
    cv2.rectangle(img, (0, 0), (W - 1, Hh - 1), (120, 120, 120), 2)

    px = lambda x, y: (int(np.clip(x * scale, 0, W - 1)), int(np.clip(y * scale, 0, Hh - 1)))
    for cam, c in cfg["cameras"].items():                         # cameras
        if c.get("position"):
            p = px(*c["position"])
            p = (min(max(p[0], 12), W - 13), min(max(p[1], 12), Hh - 13))          # keep the marker fully visible
            cv2.circle(img, p, 9, (70, 70, 70), -1, cv2.LINE_AA)
            cv2.circle(img, p, 9, (255, 255, 255), 1, cv2.LINE_AA)
            on_right = p[0] > W * 0.8
            _label(img, cam, (p[0] - 12 - 8 * len(cam) if on_right else p[0] + 14, p[1] + 5), (60, 60, 60), 0.5)

    if plan:
        for k, hop in enumerate(plan["hops"]):                     # sightings
            col = PALETTE[k % len(PALETTE)]
            pts = [px(p["x"], p["y"]) for p in hop["points"]]
            if len(pts) > 1:
                cv2.polylines(img, [np.int32(pts)], False, col, 3, cv2.LINE_AA)
                cv2.arrowedLine(img, pts[-2], pts[-1], col, 3, cv2.LINE_AA, tipLength=0.5)
            for p in pts[:1]:
                cv2.circle(img, p, 7 if hop["mode"] == "projected" else 11, col, -1 if hop["mode"] == "projected" else 2, cv2.LINE_AA)
            if pts:
                label = f"{hop['camera_id']} {hop['first_seen_timestamp']:.0f}-{hop['last_seen_timestamp']:.0f}s" + ("" if hop["mode"] == "projected" else " (approx.)")
                if hop["mode"] != "projected":
                    dy = 38                                                    # clear of the camera's own name tag
                elif len(pts) > 1 and pts[1][1] >= pts[0][1]:
                    dy = -14                                                   # path heads down -> label above its start
                else:
                    dy = 24                                                    # path heads up -> label below its start
                _label(img, label, (pts[0][0] + 10, pts[0][1] + dy), col, 0.45)
        for cn in plan["connectors"]:                              # hops between cameras / unexplained gaps
            a, b = px(cn["x1"], cn["y1"]), px(cn["x2"], cn["y2"])
            if cn["unexplained"]:
                col = RED if cn["severity"] == "critical" else AMBER
                _dashed(img, a, b, col, 3)
                mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                _label(img, f"{cn['gap_seconds']:.0f}s unseen", (mid[0] + 8, mid[1] - 8), col, 0.55, 2)
            else:
                cv2.line(img, a, b, (150, 150, 150), 1, cv2.LINE_AA)
    if title:
        _label(img, title, (10, 24), (40, 40, 40), 0.6, 2)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "check":
        rep = calibration_report(load_config(sys.argv[2]))
        for cam, r in rep.items():
            print(f"{cam}: " + ("not calibrated (pinned at camera position)" if not r["calibrated"] else
                              f"rms error {r['rms_error']} (tolerance {r['tolerance']}) -> {'OK' if r['ok'] else 'RE-CHECK YOUR POINTS'}"))
    else:
        print("usage: python -m floorplan.core check floorplan.json")
