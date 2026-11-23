"""Run: pytest -q vehicle_intel   (OCR test is skipped if tesseract/pytesseract is missing)"""
import cv2
import numpy as np
from vehicle_intel.core import (TrackAggregator, analyze_vehicle, classify_color, find_plate_region,
                                normalize_plate, plate_distance, PlateReader)


def solid(bgr, size=(120, 200), noise=6, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((*size, 3), np.int16) + np.array(bgr, np.int16)
    return np.clip(img + rng.integers(-noise, noise + 1, img.shape), 0, 255).astype(np.uint8)


def test_colors():
    cases = {"red": (30, 30, 200), "blue": (200, 60, 30), "green": (40, 160, 50), "white": (235, 235, 235),
             "black": (20, 20, 20), "yellow": (30, 220, 230), "silver": (165, 165, 165), "gray": (90, 90, 90)}
    for want, bgr in cases.items():
        got, share = classify_color(solid(bgr))
        assert got == want, (want, got)
        assert share > 0.8


def test_color_ignores_background_border():
    img = solid((200, 60, 30))                       # blue car
    img[:, :20] = (235, 235, 235); img[:, -20:] = (235, 235, 235); img[:15] = (90, 90, 90)  # road/sky at edges
    assert classify_color(img)[0] == "blue"


def test_color_empty_input():
    assert classify_color(np.zeros((0, 0, 3), np.uint8)) == ("unknown", 0.0)


def test_normalize_plate_fixes_lookalikes():
    assert normalize_plate("MH12AB1234") == ("MH12AB1234", True)
    assert normalize_plate("MH I2 AB 1Z34") == ("MH12AB1234", True)         # I->1, Z->2 in digit slots
    assert normalize_plate("MHO1AB123S")[0] == "MH01AB1235"                  # O->0, S->5
    assert normalize_plate("M012AB1234")[0] == "MO12AB1234"                  # 0->O in state slot
    assert normalize_plate("XX12AB1234")[1] is False                         # not a real state code
    assert normalize_plate("ABC")[1] is False


def test_plate_distance():
    assert plate_distance("MH12AB1234", "MH12AB1234") == 0
    assert plate_distance("MH12AB1234", "MH12AB1284") == 1
    assert plate_distance("", "ABC") == 3


def test_aggregator_majority_vote():
    a = TrackAggregator("CAM_1", 7)
    reads = [("MH12AB1234", True, .9), ("MH12AB1234", True, .8), ("MH12AB1284", True, .3), ("", False, 0)]
    for i, (p, v, c) in enumerate(reads):
        a.add(float(i), {"type": "car" if i != 2 else "truck", "color": "red", "color_share": .9,
                         "plate": p, "plate_valid": v, "plate_conf": c})
    r = a.result()
    assert (r["vehicle_type"], r["color"], r["plate"], r["plate_valid"]) == ("car", "red", "MH12AB1234", True)
    assert r["first_seen_timestamp"] == 0.0 and r["last_seen_timestamp"] == 3.0 and r["num_frames"] == 4


def test_aggregator_valid_plate_beats_confident_garbage():
    a = TrackAggregator("C", 1)
    a.add(0, {"plate": "ZZZZZZZZ", "plate_valid": False, "plate_conf": .99})
    a.add(1, {"plate": "MH12AB1234", "plate_valid": True, "plate_conf": .2})
    assert a.result()["plate"] == "MH12AB1234"


def _fake_car_with_plate(text="MH12AB1234"):
    car = solid((30, 30, 200), size=(300, 450))                    # red body
    cv2.rectangle(car, (110, 215), (340, 275), (255, 255, 255), -1)  # plate
    cv2.rectangle(car, (110, 215), (340, 275), (0, 0, 0), 2)
    cv2.putText(car, text, (120, 256), cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 0, 0), 3, cv2.LINE_AA)
    return car


def test_plate_region_found_and_ocr_end_to_end():
    import pytest
    pytest.importorskip("pytesseract")
    try:
        reader = PlateReader("tesseract")
    except Exception:
        pytest.skip("tesseract not available")
    car = _fake_car_with_plate()
    box = find_plate_region(car)
    assert box is not None
    x, y, w, h = box
    assert 80 < x < 140 and 190 < y < 240                        # lands on the plate
    res = analyze_vehicle(car, "car", reader)
    assert res["color"] == "red"
    assert res["plate"] == "MH12AB1234" and res["plate_valid"]
