"""Run: python -m floorplan.test_floorplan   (or pytest -q floorplan)"""
import unittest

import cv2
import numpy as np

from .core import (calibration_report, compute_homography, foot_point, journey_on_plan, load_config, project_points, render_png,
                   track_floor_path)

# Ground truth: a camera looking down a 20 x 12 room (far wall narrower in the picture)
IMG_QUAD = np.float32([[220, 100], [420, 100], [620, 340], [20, 340]])
FLOOR_RECT = np.float32([[0, 0], [20, 0], [20, 12], [0, 12]])
H_TRUE = cv2.getPerspectiveTransform(IMG_QUAD, FLOOR_RECT)                 # image -> floor
H_INV = np.linalg.inv(H_TRUE)                                              # floor -> image


def to_image(floor_xy):
    return project_points(H_INV, [floor_xy])[0]


def calib(extra_noise=0.0, n=6, seed=0):
    rng = np.random.default_rng(seed)
    floor = [[2, 1], [18, 1], [18, 11], [2, 11], [10, 6], [10, 1]][:n]
    img = [list(to_image(p) + rng.normal(0, extra_noise, 2)) for p in floor]
    return {"image_points": img, "floor_points": floor}


def cfg(**over):
    c = {"width": 20.0, "height": 12.0, "scale": 40, "cameras": {
        "cam1": {"position": [10, 0.2], "calibration": calib()},
        "cam2": {"position": [19.5, 6]},                                    # not calibrated
    }}
    c.update(over)
    return c


def walking_obs(cam, tid, start, end, n=40, t0=0.0, dt=0.25):
    """A person walking in a straight line on the floor; bbox bottom-centre = their feet in the image."""
    obs = []
    for i in range(n):
        f = i / (n - 1)
        fx, fy = start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f
        u, v = to_image((fx, fy))
        obs.append({"camera_id": cam, "track_id": tid, "frame": i, "timestamp": t0 + i * dt,
                    "bbox_xyxy": [int(u - 15), int(v - 90), int(u + 15), int(v)]})
    return obs


class FloorplanTests(unittest.TestCase):
    def test_homography_recovers_floor_positions(self):
        H = compute_homography(**calib())
        for p in [(3, 2), (10, 6), (17, 9), (5, 10.5)]:
            got = project_points(H, [to_image(p)])[0]
            self.assertLess(np.linalg.norm(got - np.float32(p)), 0.01, p)

    def test_homography_survives_click_noise_and_an_outlier(self):
        c = calib(extra_noise=1.5, n=6)
        c["image_points"].append([5, 5]); c["floor_points"].append([19, 11])      # a wildly wrong 7th pair
        H = compute_homography(**c)                                                # RANSAC (>4 points) ignores it
        got = project_points(H, [to_image((10, 6))])[0]
        self.assertLess(np.linalg.norm(got - np.float32([10, 6])), 0.25)

    def test_needs_four_points(self):
        with self.assertRaises(ValueError):
            compute_homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])

    def test_calibration_report_flags_bad_clicks(self):
        good = calibration_report(cfg())
        self.assertTrue(good["cam1"]["calibrated"] and good["cam1"]["ok"])
        self.assertFalse(good["cam2"]["calibrated"])
        bad = cfg(); bad["cameras"]["cam1"]["calibration"] = calib(extra_noise=40, n=4, seed=3)
        # 4 exact points always fit exactly, so check a 6-point set with one bad pair instead
        c6 = calib(n=6); c6["image_points"][4] = [30, 40]
        bad["cameras"]["cam1"]["calibration"] = c6
        self.assertFalse(calibration_report(bad)["cam1"]["ok"])

    def test_track_path_matches_ground_truth(self):
        H = compute_homography(**calib())
        obs = walking_obs("cam1", 7, (3, 2), (15, 9))
        path = track_floor_path(obs, "cam1", 7, H)
        self.assertEqual((path[0]["t"], path[-1]["t"]), (0.0, 9.75))
        self.assertLess(np.hypot(path[0]["x"] - 3, path[0]["y"] - 2), 0.1)
        self.assertLess(np.hypot(path[-1]["x"] - 15, path[-1]["y"] - 9), 0.1)
        mid = path[len(path) // 2]
        self.assertLess(abs(mid["y"] - (2 + (9 - 2) * (mid["x"] - 3) / 12)), 0.1)      # stays on the straight line
        self.assertEqual(track_floor_path(obs, "cam1", 99, H), [])
        self.assertLessEqual(len(track_floor_path(obs, "cam1", 7, H, max_points=10)), 10)

    def test_uses_feet_not_head(self):
        self.assertEqual(foot_point([10, 20, 30, 100]), (20.0, 100.0))

    def test_smoothing_removes_a_detector_glitch(self):
        H = compute_homography(**calib())
        obs = walking_obs("cam1", 1, (4, 3), (8, 3), n=30)
        u, v = to_image((19, 11)); obs[15]["bbox_xyxy"] = [int(u - 15), int(v - 90), int(u + 15), int(v)]     # one wild box
        pts = track_floor_path(obs, "cam1", 1, H, max_points=30)
        self.assertTrue(all(p["x"] < 9 for p in pts))

    def test_journey_with_uncalibrated_camera_and_gap(self):
        obs = walking_obs("cam1", 1, (3, 2), (8, 5), n=20, t0=0)
        journey = [{"camera_id": "cam1", "track_id": 1, "first_seen_timestamp": 0.0, "last_seen_timestamp": 4.75},
                   {"camera_id": "cam2", "track_id": 4, "first_seen_timestamp": 80.0, "last_seen_timestamp": 90.0}]
        gaps = [{"from_camera": "cam1", "to_camera": "cam2", "last_seen": 4.75, "next_seen": 80.0, "gap_seconds": 75.25, "severity": "critical"}]
        plan = journey_on_plan(journey, gaps, cfg(), obs)
        self.assertEqual([h["mode"] for h in plan["hops"]], ["projected", "camera_position"])
        self.assertEqual(plan["hops"][1]["points"], [{"t": 80.0, "x": 19.5, "y": 6}])
        self.assertEqual(plan["hops"][0]["out_of_bounds_ratio"], 0.0)
        cn = plan["connectors"][0]
        self.assertTrue(cn["unexplained"] and cn["severity"] == "critical" and cn["gap_seconds"] == 75.25)
        self.assertEqual((cn["x2"], cn["y2"]), (19.5, 6))

    def test_unknown_camera_is_unmapped_not_a_crash(self):
        plan = journey_on_plan([{"camera_id": "camX", "track_id": 1, "first_seen_timestamp": 0, "last_seen_timestamp": 1}], [], cfg(), [])
        self.assertEqual(plan["hops"][0]["mode"], "unmapped")
        self.assertEqual(plan["connectors"], [])

    def test_render_png(self):
        obs = walking_obs("cam1", 1, (3, 2), (8, 5), n=20)
        journey = [{"camera_id": "cam1", "track_id": 1, "first_seen_timestamp": 0.0, "last_seen_timestamp": 4.75},
                   {"camera_id": "cam2", "track_id": 4, "first_seen_timestamp": 80.0, "last_seen_timestamp": 90.0}]
        gaps = [{"from_camera": "cam1", "to_camera": "cam2", "last_seen": 4.75, "next_seen": 80.0, "gap_seconds": 75.25, "severity": "critical"}]
        png = render_png(cfg(), journey_on_plan(journey, gaps, cfg(), obs), title="test")
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape, (480, 800, 3))
        red = (np.abs(img.astype(int) - np.array([50, 50, 230])).sum(axis=2) < 40).sum()
        self.assertGreater(red, 50)                                                  # the dashed critical gap is drawn
        cam_px = img[int(0.2 * 40), int(10 * 40)]                                   # cam1 marker is dark, not background
        self.assertLess(int(cam_px.sum()), 400)
        blank = cv2.imdecode(np.frombuffer(render_png(cfg()), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual((np.abs(blank.astype(int) - np.array([50, 50, 230])).sum(axis=2) < 40).sum(), 0)   # no plan -> no red



class HelperTests(unittest.TestCase):
    def test_example_config_is_valid_and_calibration_ok(self):
        from pathlib import Path
        import tempfile
        cfg = load_config(Path(__file__).parent / "floorplan.example.json")
        rep = calibration_report(cfg)
        self.assertTrue(rep["cam1"]["calibrated"] and rep["cam1"]["ok"], rep)
        self.assertFalse(rep["cam2"]["calibrated"])
        png = render_png(cfg, None, tempfile.mkdtemp())            # background image missing -> falls back to a grid
        self.assertGreater(len(png), 1000)

    def test_grid_still_labels_a_frame(self):
        import tempfile
        import numpy as np
        import cv2
        from pathlib import Path
        from .grid_still import grid_still
        d = Path(tempfile.mkdtemp())
        w = cv2.VideoWriter(str(d / "v.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
        for _ in range(3): w.write(np.full((240, 320, 3), 40, np.uint8))
        w.release()
        self.assertEqual(grid_still(d / "v.avi", d / "g.png", step=100), (320, 240))
        img = cv2.imread(str(d / "g.png"))
        self.assertEqual(img.shape, (240, 320, 3))
        self.assertTrue((img[:, 100] == (0, 255, 255)).any())       # a yellow grid line at x=100
        with self.assertRaises(FileNotFoundError):
            grid_still(d / "missing.avi", d / "x.png")

if __name__ == "__main__":
    unittest.main(verbosity=2)
