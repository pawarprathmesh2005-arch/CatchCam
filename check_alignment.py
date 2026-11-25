"""Why does the video show boxes on empty ground (or no box on a person)?  Run:  python check_alignment.py cam1 12:38:26 [data_folder]"""
import json, sys
from pathlib import Path

import cv2

cam, clock = sys.argv[1], sys.argv[2]
d = Path(sys.argv[3] if len(sys.argv) > 3 else ".")
h, m, s = [float(x) for x in clock.split(":")]
t = h * 3600 + m * 60 + s
offs = json.loads((d / "time_offsets.json").read_text()) if (d / "time_offsets.json").exists() else {}
vids = json.loads((d / "camera_videos.json").read_text()) if (d / "camera_videos.json").exists() else {}
print("start time saved for", cam, ":", offs.get(cam, "NONE (clock starts at 0)"))
vp = d / vids.get(cam, f"videos/{cam}.mp4")
cap = cv2.VideoCapture(str(vp))
fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("video:", vp, "| exists:", vp.exists(), "| fps", round(fps, 2), "| length", round(n / fps, 1) if fps else "?", "s")
rows = json.loads((d / "outputs" / f"{cam}_observations.json").read_text())
ts = [r["timestamp"] for r in rows]
print("observations:", len(rows), "| tracks", len({r['track_id'] for r in rows}), "| times", min(ts), "to", max(ts))
print("expected video seconds at that clock:", round(t - offs.get(cam, 0), 1))
near = [r for r in rows if abs(r["timestamp"] - t) < 0.5]
print("boxes within 0.5 s of", clock, ":", sorted({(r["track_id"], tuple(round(x) for x in r["bbox_xyxy"])) for r in near}))
