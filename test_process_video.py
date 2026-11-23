"""
Tests process_video.run() end to end WITHOUT needing YOLO weights: a tiny fake `ultralytics.YOLO` feeds synthetic frames
(a red car with a readable plate, and a white bus) through the real pipeline: colour vote, plate OCR + cleaning,
per-track voting, JSON output, time offset.   Run:  python -m vehicle_intel.test_process_video
(Needs opencv + numpy + an OCR backend: pytesseract+tesseract, or easyocr.)
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

from .test_vehicle_intel import _fake_car_with_plate, solid


class _T:                                     # minimal torch-tensor look-alike
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def int(self): return _T(self.a.astype(int))
    def tolist(self): return self.a.tolist()


def _install_fake_ultralytics(frames_per_track=12):
    car = _fake_car_with_plate()                              # 300 x 450 red car with plate
    bus = solid((235, 235, 235), size=(260, 400))             # plain white bus, no plate

    class FakeYOLO:
        def __init__(self, weights): self.weights = weights
        def track(self, source, stream=True, **kw):
            assert kw["classes"] == [2, 3, 5, 7] and kw["tracker"] == "bytetrack.yaml"
            for i in range(frames_per_track):
                frame = np.full((720, 1280, 3), 90, np.uint8)
                frame[100:400, 100:550] = car
                frame[300:560, 700:1100] = bus
                boxes = np.array([[100, 100, 550, 400], [700, 300, 1100, 560]])
                yield types.SimpleNamespace(orig_img=frame, boxes=types.SimpleNamespace(
                    xyxy=_T(boxes), id=_T([1, 2]), cls=_T([2, 5])))

    mod = types.ModuleType("ultralytics"); mod.YOLO = FakeYOLO
    sys.modules["ultralytics"] = mod


def _make_video(path, fps=10, n=12):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for _ in range(n): w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()


class ProcessVideoTest(unittest.TestCase):
    def setUp(self):
        _install_fake_ultralytics()
        self.tmp = Path(tempfile.mkdtemp())
        self.video = self.tmp / "cam1.avi"; _make_video(self.video)

    def test_full_pipeline(self):
        from . import process_video
        try:
            out = process_video.run(str(self.video), "cam1", str(self.tmp / "outputs"), every=1, time_offset=100.0)
        except ImportError:
            self.skipTest("no OCR backend (install pytesseract+tesseract or easyocr)")
        rows = {r["track_id"]: r for r in json.loads(Path(out).read_text())}
        self.assertEqual(Path(out).name, "cam1_vehicles.json")
        car, bus = rows[1], rows[2]
        self.assertEqual((car["vehicle_type"], car["color"]), ("car", "red"))
        self.assertEqual(car["plate"], "MH12AB1234"); self.assertTrue(car["plate_valid"])
        self.assertEqual((bus["vehicle_type"], bus["color"], bus["plate"]), ("bus", "white", ""))
        self.assertEqual(car["num_frames"], 12)
        self.assertAlmostEqual(car["first_seen_timestamp"], 100.0)           # time_offset applied, same clock as track.py
        self.assertAlmostEqual(car["last_seen_timestamp"], 100.0 + 11 / 10)
        self.assertEqual(car["camera_id"], "cam1")

    def test_every_n_frames_sampling(self):
        from . import process_video
        try:
            out = process_video.run(str(self.video), "cam1", str(self.tmp / "o2"), every=4)
        except ImportError:
            self.skipTest("no OCR backend")
        car = {r["track_id"]: r for r in json.loads(Path(out).read_text())}[1]
        self.assertEqual(car["num_frames"], 3)                                # frames 0, 4, 8


if __name__ == "__main__":
    unittest.main(verbosity=2)
