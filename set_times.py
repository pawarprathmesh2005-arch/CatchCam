"""Set when each camera's video starts, for footage that is already processed. Works on its own (needs nothing else from CatchCam).

    python -m webapp.set_times --data demo_data cam1=12:38:22 cam2=12:46:28
"""
import argparse
import json
from pathlib import Path

KEYS = ("timestamp", "first_seen_timestamp", "last_seen_timestamp")


def parse(text):
    parts = [float(x) for x in text.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts[-3:]
    return h * 3600 + m * 60 + s


def set_start(data_dir, cam, start):
    d = Path(data_dir)
    offs_file = d / "time_offsets.json"
    try:
        offs = json.loads(offs_file.read_text(encoding="utf-8"))
    except Exception:
        offs = {}
    delta = float(start) - float(offs.get(cam, 0.0))
    for f in (d / "outputs").glob(f"{cam}_*.json"):
        try:
            rows = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            if isinstance(r, dict):
                for k in KEYS:
                    if isinstance(r.get(k), (int, float)):
                        r[k] = round(r[k] + delta, 3)
        f.write_text(json.dumps(rows), encoding="utf-8")
    offs[cam] = float(start)
    offs_file.write_text(json.dumps(offs, indent=1), encoding="utf-8")
    return delta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("times", nargs="+", help="cam1=12:38:22 ...")
    a = ap.parse_args(argv)
    for item in a.times:
        cam, _, t = item.partition("=")
        try:
            start = parse(t)
        except ValueError:
            print(f"skipped {item}: write a real time like cam3=12:50:00")
            continue
        print(cam.strip(), "starts", t, "(shifted", set_start(a.data, cam.strip(), start), "s)")


if __name__ == "__main__":
    main()
