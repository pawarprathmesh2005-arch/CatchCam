"""Run: python -m person_attrs.test_person_attrs   (or pytest -q person_attrs)"""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .core import (scene_levels, analyze_person, color_similarity, match_agreement, parse_description, search_people, score_track)
from .extract import extract


def person_image(upper_bgr, lower_bgr, h=160, w=64, seed=0):
    """Synthetic standing person: skin head, coloured torso, coloured legs, dark shoes, noisy."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.int16)
    img[:int(h * .16)] = (140, 170, 210)                       # head (skin-ish)
    img[int(h * .16):int(h * .52)] = upper_bgr
    img[int(h * .52):int(h * .94)] = lower_bgr
    img[int(h * .94):] = (30, 30, 30)
    return np.clip(img + rng.integers(-10, 11, img.shape), 0, 255).astype(np.uint8)


RED, BLUE, GREEN, BLACK, WHITE = (30, 30, 200), (200, 60, 30), (40, 160, 50), (25, 25, 25), (235, 235, 235)


class AttrTests(unittest.TestCase):
    def test_colours_of_a_synthetic_person(self):
        r = analyze_person(person_image(RED, BLUE))
        self.assertEqual((r["upper"][0], r["lower"][0]), ("red", "blue"))
        r = analyze_person(person_image(GREEN, BLACK))
        self.assertEqual((r["upper"][0], r["lower"][0]), ("green", "black"))

    def test_too_small_crop_is_skipped(self):
        self.assertEqual(analyze_person(np.zeros((30, 10, 3), np.uint8))["upper"], (None, 0.0))

    def test_parse_description(self):
        p = parse_description
        self.assertEqual(p("person wearing a red jacket and blue jeans"), {"upper": "red", "lower": "blue", "any": None})
        self.assertEqual(p("jacket in red")["upper"], "red")
        self.assertEqual(p("his shirt is blue")["upper"], "blue")
        self.assertEqual(p("black trousers")["lower"], "black")
        self.assertEqual(p("someone in green"), {"upper": None, "lower": None, "any": "green"})
        self.assertEqual(p("grey hoodie")["upper"], "gray")                 # alias
        self.assertEqual(p("navy shirt and maroon skirt"), {"upper": "blue", "lower": "red", "any": None})
        self.assertFalse(any(p("a tall man").values()))

    def test_search_ranks_exact_over_neighbour(self):
        rows = [dict(camera_id="c", track_id=1, upper_color="red", upper_conf=.9, lower_color="blue", lower_conf=.9, first_seen_timestamp=5),
                dict(camera_id="c", track_id=2, upper_color="orange", upper_conf=.9, lower_color="blue", lower_conf=.9, first_seen_timestamp=1),
                dict(camera_id="c", track_id=3, upper_color="green", upper_conf=.9, lower_color="blue", lower_conf=.9, first_seen_timestamp=2)]
        hits = search_people(rows, "red jacket")
        self.assertEqual([h["track_id"] for h in hits], [1, 2])             # green excluded, orange partial credit
        self.assertGreater(hits[0]["score"], hits[1]["score"])
        both = search_people(rows, "red jacket and blue jeans")             # partial matches are kept but rank lower
        self.assertEqual([h["track_id"] for h in both], [1, 2, 3])
        self.assertTrue(both[0]["score"] > both[1]["score"] > both[2]["score"])
        self.assertEqual(search_people(rows, "tall person"), [])

    def test_color_similarity(self):
        self.assertEqual(color_similarity("red", "red"), 1.0)
        self.assertEqual(color_similarity("white", "silver"), 0.5)
        self.assertEqual(color_similarity("red", "blue"), 0.0)
        self.assertEqual(color_similarity(None, "red"), 0.0)

    def test_match_agreement(self):
        a = dict(upper_color="red", upper_conf=.9, lower_color="blue", lower_conf=.9)
        self.assertEqual(match_agreement(a, dict(a))["verdict"], "consistent")
        diff = dict(upper_color="green", upper_conf=.9, lower_color="black", lower_conf=.9)
        self.assertEqual(match_agreement(a, diff)["verdict"], "conflict")
        # only the jacket differs (taken off / lighting): NOT a conflict
        jacket_off = dict(upper_color="white", upper_conf=.9, lower_color="blue", lower_conf=.9)
        self.assertEqual(match_agreement(a, jacket_off)["verdict"], "consistent")
        # differ but unsure -> unknown, not conflict
        shaky = dict(upper_color="green", upper_conf=.3, lower_color="black", lower_conf=.3)
        self.assertEqual(match_agreement(a, shaky)["verdict"], "unknown")
        self.assertEqual(match_agreement(a, None)["verdict"], "unknown")

    def test_extract_from_video(self):
        tmp = Path(tempfile.mkdtemp())
        people = {1: (RED, BLUE), 2: (GREEN, BLACK), 3: (WHITE, BLUE)}
        video = tmp / "cam1.avi"
        w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (400, 240))
        obs = []
        for f in range(20):
            frame = np.full((240, 400, 3), 100, np.uint8)
            for tid, (u, l) in people.items():
                x = 20 + (tid - 1) * 120 + f
                frame[40:200, x:x + 64] = person_image(u, l, seed=f + tid)
                obs.append({"camera_id": "cam1", "track_id": tid, "frame": f, "timestamp": f / 10, "class": "person",
                            "bbox_xyxy": [x, 40, x + 64, 200]})
            w.write(frame)
        w.release()
        obs.append({"camera_id": "cam1", "track_id": 99, "frame": 3, "timestamp": 0.3, "class": "car", "bbox_xyxy": [0, 0, 50, 50]})
        (tmp / "obs.json").write_text(json.dumps(obs))
        out = extract(video, tmp / "obs.json", "cam1", tmp / "outputs")
        rows = {r["track_id"]: r for r in json.loads(Path(out).read_text())}
        self.assertEqual(set(rows), {1, 2, 3})                               # the car is not a person
        self.assertEqual((rows[1]["upper_color"], rows[1]["lower_color"]), ("red", "blue"))
        self.assertEqual((rows[2]["upper_color"], rows[2]["lower_color"]), ("green", "black"))
        self.assertEqual((rows[3]["upper_color"], rows[3]["lower_color"]), ("white", "blue"))
        self.assertEqual((rows[1]["first_seen_timestamp"], rows[1]["last_seen_timestamp"]), (0.0, 1.9))
        self.assertTrue(all(r["upper_conf"] > .8 for r in rows.values()))

    def test_extract_accepts_decimal_box_coordinates(self):
        """Regression: real detectors (YOLO) write pixel positions as decimals like 173.6; slicing a frame with one raised TypeError."""
        tmp = Path(tempfile.mkdtemp())
        video = tmp / "cam1.avi"
        w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (400, 240))
        obs = []
        for f in range(12):
            frame = np.full((240, 400, 3), 100, np.uint8)
            frame[40:200, 30:94] = person_image(RED, BLUE, seed=f)
            obs.append({"camera_id": "cam1", "track_id": 1, "frame": f, "timestamp": f / 10, "class": "person",
                        "confidence": 0.9, "bbox_xyxy": [29.6, 39.7, 94.2, 200.4]})
            w.write(frame)
        w.release()
        (tmp / "obs.json").write_text(json.dumps(obs))
        out = extract(video, tmp / "obs.json", "cam1", tmp / "outputs")
        row = json.loads(Path(out).read_text())[0]
        self.assertEqual((row["upper_color"], row["lower_color"]), ("red", "blue"))

    # ---- washed-out cameras (found on real lobby footage: a black T-shirt measured ~100/255, so the fixed rule called it gray)
    @staticmethod
    def _washed_scene(frames=6):
        """Bright marble floor (250) with a dark carpet strip (75) on the left: the darkest the camera ever shows is ~75, not ~10."""
        out = []
        for k in range(frames):
            f = np.full((180, 320, 3), 250, np.uint8)
            f[:, :40] = 75
            out.append(f)
        return out

    def test_scene_levels_find_the_cameras_own_black_and_white(self):
        black, white = scene_levels(self._washed_scene())
        self.assertTrue(60 <= black <= 90 and white >= 240, (black, white))
        self.assertEqual(scene_levels([]), (0, 255))
        black, white = scene_levels([np.full((90, 160, 3), 128, np.uint8)] * 6)      # nothing dark in the scene: never trust a high black level
        self.assertLessEqual(black, 90)

    def test_black_shirt_on_an_overexposed_camera_is_still_black(self):
        lv = scene_levels(self._washed_scene())
        lifted_black, lifted_jeans = (100, 100, 105), (120, 100, 90)               # what 'black' and dark jeans look like on this camera
        crop = person_image(lifted_black, lifted_jeans)
        self.assertEqual(analyze_person(crop)["upper"][0], "gray")                 # without levels: the old (wrong) answer, kept for cameras we know nothing about
        self.assertEqual(analyze_person(crop, lv)["upper"][0], "black")            # with the camera's levels: right
        grey_shirt = analyze_person(person_image((165, 165, 165), BLUE), lv)["upper"][0]
        self.assertIn(grey_shirt, ("silver", "gray"))                              # a genuinely grey shirt is not turned black

    def test_pale_clothing_colour_and_white_floor_around_the_person(self):
        lv = (70, 254)
        pale_green = person_image((120, 175, 125), BLACK)                          # washed-out green, saturation far below 70
        self.assertEqual(analyze_person(pale_green, lv)["upper"][0], "green")
        crop = person_image(RED, BLUE)
        crop[:, :14] = 250                                                          # white floor beside the body, inside the box
        crop[:, -14:] = 250
        self.assertEqual(analyze_person(crop, lv)["upper"][0], "red")

    def test_extract_uses_the_cameras_levels(self):
        tmp = Path(tempfile.mkdtemp())
        video = tmp / "lobby.avi"
        w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (400, 240))
        obs = []
        for f in range(30):
            frame = np.full((240, 400, 3), 250, np.uint8)
            frame[:, :40] = 75                                                      # dark carpet: shows where black is on this camera
            if 5 <= f < 25:
                x = 120 + f
                frame[40:200, x:x + 64] = person_image((100, 100, 105), (120, 100, 90), seed=f)
                obs.append({"camera_id": "lobby", "track_id": 1, "frame": f, "timestamp": f / 10, "class": "person",
                            "confidence": .9, "bbox_xyxy": [x, 40, x + 64, 200]})
            w.write(frame)
        w.release()
        (tmp / "obs.json").write_text(json.dumps(obs))
        out = extract(video, tmp / "obs.json", "lobby", tmp / "outputs")
        row = json.loads(Path(out).read_text())[0]
        self.assertEqual(row["upper_color"], "black")
        self.assertTrue(search_people([row], "person in black tshirt"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
