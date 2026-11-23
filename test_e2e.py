"""
End-to-end API test for the new CatchCam modules (missing-camera detection + vehicle intelligence).

Run from your project folder (needs fastapi + httpx, which you already have if the API runs):
    pytest -q test_e2e.py          # or:  python test_e2e.py

It builds its own tiny synthetic dataset in a temp folder, points CATCHCAM_DATA_DIR at it, loads the REAL api.py
(including the routes_*.py auto-loader) and calls every endpoint through FastAPI's TestClient.
It never touches your real outputs/ folder.
"""
import json
import os
import tempfile
import time
from pathlib import Path

DATA = Path(tempfile.mkdtemp(prefix="catchcam_test_"))
os.environ["CATCHCAM_DATA_DIR"] = str(DATA)          # must be set BEFORE importing api / routes

(DATA / "outputs").mkdir()
(DATA / "matches").mkdir()


def _obs(cam, tid, t0, t1, fps=10):
    n = int((t1 - t0) * 2) + 1
    return [{"camera_id": cam, "track_id": tid, "frame": int((t0 + i * 0.5) * fps), "timestamp": round(t0 + i * 0.5, 2),
             "class": "person", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 20]} for i in range(n)]


# one person: cam1 (0-10 s) -> cam2 (18-30 s) -> [70 s unexplained] -> cam3 (100-110 s); plus an unrelated person on cam1
(DATA / "outputs" / "cam1_observations.json").write_text(json.dumps(_obs("cam1", 1, 0, 10) + _obs("cam1", 9, 50, 55)))
(DATA / "outputs" / "cam2_observations.json").write_text(json.dumps(_obs("cam2", 4, 18, 30)))
(DATA / "outputs" / "cam3_observations.json").write_text(json.dumps(_obs("cam3", 2, 100, 110)))
(DATA / "matches" / "cross_camera_matches.json").write_text(json.dumps([
    {"camera_a": "cam1", "track_a": 1, "camera_b": "cam2", "track_b": 4, "similarity": 0.90},
    {"camera_a": "cam2", "track_a": 4, "camera_b": "cam3", "track_b": 2, "similarity": 0.85}]))
def refresh_live_files():
    """cam1 live (just now), cam2 silent for 60 s. Called again right before each test (see conftest.py): in a long full-suite run
    the files made at import time would otherwise be 'old' by the time the health tests run."""
    for cam, age in (("cam1", 0), ("cam2", 60)):
        p = DATA / "outputs" / f"{cam}_observations_live.json"
        p.write_text("[]")
        t = time.time() - age
        os.utime(p, (t, t))


refresh_live_files()
(DATA / "outputs" / "cam1_vehicles.json").write_text(json.dumps([
    {"camera_id": "cam1", "track_id": 7, "vehicle_type": "car", "color": "red", "plate": "MH12AB1234", "plate_valid": True,
     "plate_support": 1.0, "first_seen_timestamp": 5.0, "last_seen_timestamp": 12.0, "num_frames": 9},
    {"camera_id": "cam3", "track_id": 3, "vehicle_type": "car", "color": "red", "plate": "MH12AB1284", "plate_valid": True,
     "plate_support": 0.8, "first_seen_timestamp": 95.0, "last_seen_timestamp": 99.0, "num_frames": 5},
    {"camera_id": "cam2", "track_id": 1, "vehicle_type": "bus", "color": "white", "plate": "", "plate_valid": False,
     "plate_support": 0, "first_seen_timestamp": 1.0, "last_seen_timestamp": 3.0, "num_frames": 3}]))

from fastapi.testclient import TestClient  # noqa: E402
import api  # noqa: E402  (the real api.py)

c = TestClient(api.app)


# ---------------------------------------------------------------- loader + untouched core endpoints
def test_modules_loaded():
    assert {"routes_health", "routes_vehicles"} <= set(c.get("/modules").json()["loaded"])


def test_core_endpoints_still_work():
    assert c.get("/health").json()["observation_files_found"] == 3
    cams = {x["camera_id"]: x["num_tracks"] for x in c.get("/cameras").json()["cameras"]}
    assert cams == {"cam1": 2, "cam2": 1, "cam3": 1}
    fwd = c.get("/journey", params={"camera_id": "cam1", "track_id": 1, "direction": "forward"}).json()["journey"]
    assert [h["camera_id"] for h in fwd] == ["cam1", "cam2", "cam3"]
    back = c.get("/journey", params={"camera_id": "cam3", "track_id": 2, "direction": "backward"}).json()["journey"]
    assert [h["camera_id"] for h in back] == ["cam1", "cam2", "cam3"]
    assert c.get("/incident", params={"start_time": 0, "end_time": 120}).status_code == 200


# ---------------------------------------------------------------- missing-camera: coverage gaps
def test_gap_found_from_the_middle_of_a_journey():
    r = c.get("/coverage_gaps", params={"camera_id": "cam2", "track_id": 4})
    assert r.status_code == 200
    d = r.json()
    assert [h["camera_id"] for h in d["journey"]] == ["cam1", "cam2", "cam3"]      # full journey, not just the part after cam2
    assert d["journey_length"] == 3 and d["has_blind_spot"] is True and len(d["gaps"]) == 1
    g = d["gaps"][0]
    assert (g["from_camera"], g["to_camera"], g["gap_seconds"], g["severity"]) == ("cam2", "cam3", 70.0, "critical")
    assert {"from_track", "to_track", "last_seen", "next_seen", "flag"} <= set(g)   # fields the dashboard reads


def test_gap_thresholds():
    p = {"camera_id": "cam1", "track_id": 1}
    assert c.get("/coverage_gaps", params={**p, "critical_gap": 100}).json()["gaps"][0]["severity"] == "suspicious"
    assert c.get("/coverage_gaps", params={**p, "max_normal_gap": 100}).json()["has_blind_spot"] is False
    assert len(c.get("/coverage_gaps", params={**p, "max_normal_gap": 5}).json()["gaps"]) == 2   # the 8 s hop now counts too


def test_isolated_sighting_has_no_gaps():
    d = c.get("/coverage_gaps", params={"camera_id": "cam1", "track_id": 9}).json()
    assert d["journey_length"] == 1 and d["gaps"] == []


def test_gap_errors():
    assert c.get("/coverage_gaps", params={"camera_id": "cam9", "track_id": 1}).status_code == 404
    assert c.get("/coverage_gaps", params={"camera_id": "cam1"}).status_code == 422


# ---------------------------------------------------------------- missing-camera: live health
def test_camera_health():
    d = c.get("/camera_health", params={"stale_after": 10}).json()
    assert {x["camera_id"]: x["is_down"] for x in d["cameras"]} == {"cam1": False, "cam2": True}
    assert d["num_down"] == 1 and d["status"] == "checked"
    assert all("last_update_seconds_ago" in x and "status" in x for x in d["cameras"])
    assert c.get("/camera_health", params={"stale_after": 120}).json()["num_down"] == 0


# ---------------------------------------------------------------- vehicles
def test_vehicle_filters():
    def n(**p): return c.get("/vehicles", params=p).json()["count"]
    assert n() == 3
    assert n(color="red") == 2 and n(vehicle_type="bus") == 1 and n(camera_id="cam3") == 1
    assert n(plate="MH12") == 2                                   # partial plate
    assert n(plate="MH12AB1234", max_errors=0) == 1               # exact only
    assert n(plate="MH12AB1234", max_errors=1) == 2               # one OCR error tolerated (...1284)
    assert n(color="red", vehicle_type="bus") == 0
    assert c.get("/vehicles", params={"max_errors": 9}).status_code == 422


def test_vehicle_journey_has_gap():
    d = c.get("/vehicles/journey", params={"plate": "MH12AB1234"}).json()
    assert [s["camera_id"] for s in d["sightings"]] == ["cam1", "cam3"]
    assert d["gaps"][0]["gap_seconds"] == 83.0 and d["gaps"][0]["from_camera"] == "cam1"
    assert c.get("/vehicles/journey", params={"plate": "ZZ99ZZ9999"}).status_code == 404


def test_vehicle_journey_does_not_leak_query_defaults():
    """Regression: vehicle_journey calls the shared search directly; omitted filters must mean 'no filter'."""
    d = c.get("/vehicles/journey", params={"plate": "MH12", "max_errors": 0}).json()
    assert len(d["sightings"]) == 2


if __name__ == "__main__":
    import traceback
    ok = bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); ok += 1; print("PASS", name)
            except Exception:
                bad += 1; print("FAIL", name); traceback.print_exc()
    print(f"{ok} passed, {bad} failed")
    raise SystemExit(1 if bad else 0)
