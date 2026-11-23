"""
CatchCam — watchlist alerts.

A watchlist item is a standing question the system keeps checking as new footage is processed:
  plate        "MH12AB1234"          any vehicle whose plate matches (one wrong character is tolerated, OCR is imperfect)
  clothing     "red jacket"          any person whose clothing colours match
  gap          "30"                  any journey with more than 30 s unaccounted between two cameras
  camera_down  ""                    any live camera whose feed stops

Hits become alerts (alerts.json). Hits that already existed when you added the item are shown but marked "already in the footage"
and are never sent out; only hits that appear afterwards are sent to the webhook / Telegram (see notify.py).
Everything is recomputed only when the data files change, so polling this every few seconds is cheap.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from missing_camera import check_live_camera_health, find_coverage_gaps, get_full_journey
from person_attrs.core import matches_all, parse_description
from qa_engine.engine import Store
from vehicle_intel.core import normalize_plate, plate_distance

from . import notify

KINDS = ("plate", "clothing", "gap", "camera_down")
WATCH_FILE, ALERT_FILE = "watchlist.json", "alerts.json"
MAX_ITEMS, MAX_ALERTS, DOWN_AFTER = 100, 500, 10.0
_LOCK = threading.RLock()
_SIG: Dict[str, tuple] = {}
_THREADS: List[threading.Thread] = []


# ----------------------------------------------------------------------------- storage
def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else default
    except (ValueError, OSError):
        return default


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def _items(d: Path) -> List[dict]:
    return _read(d / WATCH_FILE, {"items": []}).get("items", [])


def _state(d: Path) -> dict:
    s = _read(d / ALERT_FILE, {})
    return {"alerts": s.get("alerts", []), "down": s.get("down", [])}


def _t(x) -> str:
    if x is None:
        return "?"
    m, s = divmod(float(x), 60)
    return f"{int(m)}:{int(s):02d}"


# ----------------------------------------------------------------------------- watchlist items
def _clean_plate(value: str) -> str:
    t = re.sub(r"[^A-Za-z0-9]", "", value or "").upper()
    if not 4 <= len(t) <= 11:
        raise ValueError("A plate needs 4 to 11 letters and digits, like MH12AB1234 (or the start of one, like MH12).")
    return normalize_plate(t)[0] if len(t) >= 8 else t


def add_item(data_dir, kind: str, value: str = "", label: Optional[str] = None) -> dict:
    d = Path(data_dir)
    kind, value = (kind or "").strip().lower(), (value or "").strip()
    if kind not in KINDS:
        raise ValueError(f"kind must be one of: {', '.join(KINDS)}.")
    if len(value) > 80:
        raise ValueError("That is too long (80 characters at most).")
    if kind == "plate":
        value = _clean_plate(value)
        label = label or f"Plate {value}"
    elif kind == "clothing":
        if not any(parse_description(value).values()):
            raise ValueError("Describe the clothing with a colour, like 'red jacket' or 'blue shirt and black pants'.")
        label = label or f"Person in {value}"
    elif kind == "gap":
        try:
            secs = float(value)
        except ValueError:
            raise ValueError("Give the number of seconds, like 30.")
        if not 1 <= secs <= 3600:
            raise ValueError("Seconds must be between 1 and 3600.")
        value = f"{secs:g}"
        label = label or f"Journey with more than {value}s unaccounted"
    else:
        value, label = "", label or "Any live camera goes down"
    with _LOCK:
        data = _read(d / WATCH_FILE, {"items": []})
        if len(data["items"]) >= MAX_ITEMS:
            raise ValueError(f"The watchlist is full ({MAX_ITEMS} items). Remove some first.")
        if any(i["kind"] == kind and i["value"] == value for i in data["items"]):
            raise ValueError("That is already on the watchlist.")
        item = {"id": uuid.uuid4().hex[:8], "kind": kind, "value": value, "label": label[:120], "created": time.time(),
                "created_text": time.strftime("%d %b %H:%M")}
        data["items"].append(item)
        _write(d / WATCH_FILE, data)
        st = _state(d)
        known = {a["key"] for a in st["alerts"]}
        hits = _hits(d, item, st)
        for h in hits:
            key = f"{item['id']}|{h['key']}"
            if key not in known:
                st["alerts"].append(_make_alert(item, h, historical=True))
        if kind == "camera_down":
            st["down"] = sorted(set(st["down"]) | {h["camera_id"] for h in hits})
        _save_state(d, st)
        _SIG.pop(str(d), None)
        return {**item, "existing_hits": len(hits)}


def remove_item(data_dir, item_id: str) -> bool:
    d = Path(data_dir)
    with _LOCK:
        data = _read(d / WATCH_FILE, {"items": []})
        keep = [i for i in data["items"] if i["id"] != item_id]
        if len(keep) == len(data["items"]):
            return False
        _write(d / WATCH_FILE, {"items": keep})
        _SIG.pop(str(d), None)
        return True


# ----------------------------------------------------------------------------- finding hits
def _hits(d: Path, item: dict, state: dict) -> List[dict]:
    store = Store(d)
    kind, value = item["kind"], item["value"]
    out: List[dict] = []
    if kind == "plate":
        for v in store.vehicles():
            plate = re.sub(r"[^A-Za-z0-9]", "", str(v.get("plate") or "")).upper()
            if not plate:
                continue
            exact = len(value) < 8 and value in plate
            dist = plate_distance(value, plate) if len(value) >= 8 else (0 if exact else 99)
            if dist <= 1:
                out.append({"key": f"{v['camera_id']}:{v['track_id']}", "camera_id": v["camera_id"], "track_id": v["track_id"],
                            "title": f"Plate {plate} seen on {v['camera_id']}",
                            "detail": f"{(v.get('color') or '')} {(v.get('vehicle_type') or 'vehicle')}".strip() + f", track {v['track_id']}, "
                                      f"{_t(v.get('first_seen_timestamp'))} to {_t(v.get('last_seen_timestamp'))}"
                                      + (f". One character differs from the watchlist plate {value}." if dist == 1 else ".")})
    elif kind == "clothing":
        want = parse_description(value)
        for r in store.attributes():
            if matches_all(r, want):
                out.append({"key": f"{r['camera_id']}:{r['track_id']}", "camera_id": r["camera_id"], "track_id": r["track_id"],
                            "title": f"{item['label']} seen on {r['camera_id']}",
                            "detail": f"track {r['track_id']}, {_t(r.get('first_seen_timestamp'))} to {_t(r.get('last_seen_timestamp'))}, "
                                      f"{r.get('upper_color')} top and {r.get('lower_color')} bottoms"})
    elif kind == "gap":
        G = store.graph()
        limit = float(value)
        if G is not None:
            for n in G.nodes:
                if G.in_degree(n) == 0 and G.out_degree(n) > 0:
                    a = G.nodes[n]
                    for g in find_coverage_gaps(get_full_journey(G, a["camera_id"], a["track_id"]), limit):
                        out.append({"key": f"{a['camera_id']}:{a['track_id']}|{g['from_camera']}>{g['to_camera']}",
                                    "camera_id": g["from_camera"], "track_id": g["from_track"],
                                    "title": f"{g['gap_seconds']:.0f}s unaccounted between {g['from_camera']} and {g['to_camera']}",
                                    "detail": f"last seen {_t(g['last_seen'])} on {g['from_camera']}, next seen {_t(g['next_seen'])} on {g['to_camera']}"})
    elif kind == "camera_down":
        res = check_live_camera_health(str(store.outputs), DOWN_AFTER)
        for c in res["cameras"]:
            if c["status"].startswith("DOWN"):
                out.append({"key": c["camera_id"], "camera_id": c["camera_id"], "track_id": None,
                            "title": f"{c['camera_id']} is down", "detail": f"no update for {c['last_update_seconds_ago']:.0f}s"})
    return out


def _make_alert(item: dict, hit: dict, historical: bool) -> dict:
    return {"id": uuid.uuid4().hex[:10], "key": f"{item['id']}|{hit['key']}", "item_id": item["id"], "kind": item["kind"],
            "title": hit["title"], "detail": hit["detail"], "camera_id": hit["camera_id"], "track_id": hit["track_id"],
            "created": time.time(), "created_text": time.strftime("%d %b %H:%M:%S"), "acknowledged": False,
            "historical": historical, "notification": None}


def _save_state(d: Path, st: dict) -> None:
    alerts = st["alerts"]
    if len(alerts) > MAX_ALERTS:                                           # drop the oldest acknowledged ones first
        keep_unack = [a for a in alerts if not a["acknowledged"]]
        keep_ack = [a for a in alerts if a["acknowledged"]][-max(MAX_ALERTS - len(keep_unack), 0):]
        alerts = sorted(keep_unack + keep_ack, key=lambda a: a["created"])
    _write(d / ALERT_FILE, {"alerts": alerts, "down": st["down"]})


def _signature(d: Path, items: List[dict]) -> tuple:
    files = []
    out = d / "outputs"
    for p in list(sorted(out.glob("*.json"))) if out.exists() else []:
        files.append(p)
    files += [d / "matches" / "cross_camera_matches.json", d / WATCH_FILE]
    sig = [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists()]
    if any(i["kind"] == "camera_down" for i in items):
        sig.append(("clock", int(time.time() // 5)))                       # "down" depends on time passing, not only on files
    return tuple(sig)


# ----------------------------------------------------------------------------- the scan
def scan(data_dir) -> int:
    """Looks for new hits for every watchlist item. Returns how many new alerts were raised."""
    d = Path(data_dir)
    with _LOCK:
        items = _items(d)
        sig = _signature(d, items)
        if _SIG.get(str(d)) == sig:
            return 0
        if not items:
            _SIG[str(d)] = sig
            return 0
        st = _state(d)
        known = {a["key"] for a in st["alerts"]}
        new: List[dict] = []
        down_now = set()
        for item in items:
            for h in _hits(d, item, st):
                if item["kind"] == "camera_down":
                    down_now.add(h["camera_id"])
                    if h["camera_id"] in st["down"]:
                        continue                                           # still down from before: already told
                    h = {**h, "key": f"{h['key']}@{time.time_ns()}"}     # every outage is its own event
                key = f"{item['id']}|{h['key']}"
                if key in known:
                    continue
                a = _make_alert(item, h, historical=False)
                known.add(key)
                new.append(a)
        if any(i["kind"] == "camera_down" for i in items):
            st["down"] = sorted(down_now)                                  # a camera that came back can raise a new alert when it drops again
        st["alerts"] += new
        _save_state(d, st)
        _SIG[str(d)] = _signature(d, items)
    if new:
        _send(d, new)
    return len(new)


def _send(d: Path, alerts: List[dict]) -> None:
    cfg = notify.load_config(d)
    if not any(notify.configured(cfg).values()):
        return

    def work():
        for a in alerts:
            status = notify.deliver(cfg, f"CatchCam alert: {a['title']} ({a['detail']})", {k: a[k] for k in ("id", "kind", "title", "detail", "camera_id", "track_id")})
            with _LOCK:
                st = _state(d)
                for x in st["alerts"]:
                    if x["id"] == a["id"]:
                        x["notification"] = status
                _save_state(d, st)

    t = threading.Thread(target=work, daemon=True)
    _THREADS.append(t)
    t.start()


def wait_for_notifications(timeout: float = 10.0) -> None:
    """For tests: wait until alerts have been sent out."""
    for t in list(_THREADS):
        t.join(timeout)


# ----------------------------------------------------------------------------- reading and acknowledging
def overview(data_dir) -> dict:
    d = Path(data_dir)
    scan(d)
    st = _state(d)
    items = _items(d)
    alerts = sorted(st["alerts"], key=lambda a: -a["created"])
    for a in alerts:
        a["link"] = (f"#investigate/{a['camera_id']}/{a['track_id']}" if a["track_id"] is not None else "#cameras")
    counts: Dict[str, int] = {}
    for a in st["alerts"]:
        counts[a["item_id"]] = counts.get(a["item_id"], 0) + 1
    return {"alerts": alerts[:200], "unacknowledged": sum(1 for a in st["alerts"] if not a["acknowledged"]),
            "items": [{**i, "alerts": counts.get(i["id"], 0)} for i in items], "notifications": notify.configured(notify.load_config(d))}


def acknowledge(data_dir, alert_id: Optional[str] = None, everything: bool = False) -> int:
    d = Path(data_dir)
    with _LOCK:
        st = _state(d)
        n = 0
        for a in st["alerts"]:
            if not a["acknowledged"] and (everything or a["id"] == alert_id):
                a["acknowledged"] = True
                n += 1
        _save_state(d, st)
        return n
