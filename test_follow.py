"""Follow-a-person ranking, built from the weaknesses seen on real footage (everybody scoring ~0.8, stale embeddings, no start times)."""
import json
from pathlib import Path

import numpy as np
import pytest
from webapp import core, follow, set_times


def make(tmp_path, offsets=None):
    rng = np.random.default_rng(1)
    common = rng.normal(size=64) * 3                      # what every person shares: raw cosine is high for everyone
    person = rng.normal(size=64)
    other = rng.normal(size=64)
    emb = {
        "cam1": {1: person},
        "cam2": {5: person + rng.normal(size=64) * 0.1, 6: other, 7: person},   # 7 has no observations: stale
        "cam3": {9: other + rng.normal(size=64) * 0.1},
    }
    (tmp_path / "outputs").mkdir(); (tmp_path / "embeddings").mkdir()
    cols = {("cam1", 1): ("black", "blue"), ("cam2", 5): ("gray", "blue"), ("cam2", 6): ("white", "white"), ("cam3", 9): ("white", "gray")}
    for cam, tr in emb.items():
        (tmp_path / "embeddings" / f"{cam}_embeddings.json").write_text(json.dumps(
            {"camera_id": cam, "tracks": {str(t): {"embedding": list(map(float, v + common))} for t, v in tr.items()}}))
        obs, attrs = [], []
        for t in tr:
            if (cam, t) not in cols:
                continue
            base = {"cam1": 0, "cam2": 100, "cam3": 5000}[cam] + t
            obs += [{"camera_id": cam, "track_id": t, "frame": i, "timestamp": base + i, "class": "person", "confidence": .9, "bbox_xyxy": [1, 1, 5, 9]} for i in range(5)]
            attrs.append({"camera_id": cam, "track_id": t, "upper_color": cols[(cam, t)][0], "lower_color": cols[(cam, t)][1]})
        (tmp_path / "outputs" / f"{cam}_observations.json").write_text(json.dumps(obs))
        (tmp_path / "outputs" / f"{cam}_attributes.json").write_text(json.dumps(attrs))
    if offsets:
        (tmp_path / "time_offsets.json").write_text(json.dumps(offsets))
    return tmp_path


def test_best_candidate_is_the_same_person_and_stale_tracks_are_ignored(tmp_path):
    d = make(tmp_path)
    r = follow.follow(d, "cam1", 1)
    cams = {c["camera_id"]: c["candidates"] for c in r["cameras"]}
    assert cams["cam2"][0]["track_id"] == 5 and cams["cam2"][0]["score"] > cams["cam2"][1]["score"] + 0.2
    assert 7 not in [x["track_id"] for x in cams["cam2"]]
    assert r["cameras"][0]["camera_id"] == "cam2"                      # the camera with the clearest match comes first
    assert r["missing_start_times"] == ["cam1", "cam2", "cam3"]


def test_far_apart_in_time_is_dropped_only_when_start_times_are_known(tmp_path):
    d = make(tmp_path, {"cam1": 0, "cam2": 0, "cam3": 0})
    r = follow.follow(d, "cam1", 1, max_time_gap=600)
    assert [c["camera_id"] for c in r["cameras"]] == ["cam2"] and r["missing_start_times"] == []


def test_colours_washed_out_cameras_are_not_treated_as_a_mismatch():
    assert follow.colour_score("gray", "silver") == 0.6 and follow.colour_score("black", "black") == 1
    assert follow.colour_score("black", "gray") == 0.35 and follow.colour_score("blue", "gray") == 0 and follow.colour_score(None, "blue") == 0.5


def test_route_and_errors(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import routes_web
    d = make(tmp_path)
    app = FastAPI(); app.include_router(routes_web.router)
    c = TestClient(app)
    old = routes_web.DATA_DIR; routes_web.DATA_DIR = d
    try:
        ok = c.get("/web/follow", params={"camera_id": "cam1", "track_id": 1}).json()
        no = c.get("/web/follow", params={"camera_id": "cam1", "track_id": 99})
    finally:
        routes_web.DATA_DIR = old
    assert ok["cameras"][0]["candidates"][0]["thumb_url"].startswith("/web/thumb") and no.status_code == 404


def test_set_times_command_shifts_saved_results(tmp_path):
    d = make(tmp_path)
    set_times.main(["--data", str(d), "cam1=12:38:22", "cam2=0:08:00"])
    offs = json.loads((d / "time_offsets.json").read_text())
    assert offs == {"cam1": 45502.0, "cam2": 480.0}
    assert json.loads((d / "outputs" / "cam1_observations.json").read_text())[0]["timestamp"] == 45503.0
    assert set_times.parse("12:38:22") == 45502 and set_times.parse("1:05") == 65


def test_reprocessing_keeps_the_saved_start_time(tmp_path, monkeypatch):
    """Re-running a camera must not reset its times to 0 while the saved start time still says 12:38:22 (boxes then land on empty ground)."""
    pytest.importorskip("cv2")
    from webapp import jobs, tracker
    (tmp_path / "outputs").mkdir()
    (tmp_path / "time_offsets.json").write_text(json.dumps({"cam1": 45502.0}))
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "cam1.mp4").write_bytes(b"x")
    (tmp_path / "camera_videos.json").write_text(json.dumps({"cam1": str(tmp_path / "videos" / "cam1.mp4")}))
    seen = {}

    def fake_run(video, cam, out, **kw):
        seen["t"] = kw["time_offset"]
        raise RuntimeError("stop here")
    monkeypatch.setattr(tracker, "run", fake_run)
    job = type("J", (), {"__setattr__": lambda s, k, v: None, "__getattr__": lambda s, k: None})()
    try:
        jobs.process_footage(tmp_path, "cam1", job)
    except Exception:
        pass
    assert seen.get("t") == 45502.0


def test_clip_cutter_starts_exactly_where_asked_even_when_asked_out_of_order(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from webapp.motion import ClipCutter
    p = str(tmp_path / "v.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 90))
    if not w.isOpened():
        pytest.skip("this OpenCV build cannot write mp4")
    for i in range(300):
        f = np.zeros((90, 640, 3), np.uint8); f[30:60, i:i + 10] = 255; w.write(f)
    w.release()

    def first(path):
        ok, f = cv2.VideoCapture(path).read()
        return int(np.argmax(f[:, :, 0].mean(0) > 40))
    cc = ClipCutter(p)
    for s, e in [(37, 60), (150, 170), (20, 30)]:                         # the last one goes backwards
        assert cc.clip(s, e, str(tmp_path / "c.avi")) == e - s and abs(first(str(tmp_path / "c.avi")) - s) <= 1
    cc.close()


def test_identities_join_comebacks_and_other_cameras_but_never_two_people_seen_together(tmp_path):
    from webapp import identities
    rng = np.random.default_rng(3)
    common = rng.normal(size=64) * 3
    A, B = rng.normal(size=64), rng.normal(size=64)
    spec = {("cam1", 1): (A, 0, 10, "black", "blue"), ("cam1", 4): (A, 100, 110, "black", "blue"), ("cam2", 5): (A, 30, 40, "gray", "blue"),
            ("cam1", 2): (B, 5, 15, "white", "white"), ("cam2", 6): (B, 200, 210, "white", "gray"),
            ("cam1", 3): (A, 3, 12, "black", "blue")}     # looks like A but is on screen together with track 1: a different person
    (tmp_path / "outputs").mkdir(); (tmp_path / "embeddings").mkdir()
    for cam in ("cam1", "cam2"):
        obs, attrs, embs = [], [], {}
        for (c, t), (vec, t0, t1, up, lo) in spec.items():
            if c != cam:
                continue
            obs += [{"camera_id": c, "track_id": t, "frame": i, "timestamp": t0 + i, "class": "person", "confidence": .9, "bbox_xyxy": [1, 1, 5, 9]} for i in range(t1 - t0)]
            attrs.append({"camera_id": c, "track_id": t, "upper_color": up, "lower_color": lo})
            embs[str(t)] = {"embedding": list(map(float, vec + common + rng.normal(size=64) * 0.05))}
        (tmp_path / "outputs" / f"{cam}_observations.json").write_text(json.dumps(obs))
        (tmp_path / "outputs" / f"{cam}_attributes.json").write_text(json.dumps(attrs))
        (tmp_path / "embeddings" / f"{cam}_embeddings.json").write_text(json.dumps({"camera_id": cam, "tracks": embs}))
    r = identities.build(tmp_path)
    groups = [{(m["camera_id"], m["track_id"]) for m in p["members"]} for p in r["people"]]
    assert {("cam1", 1), ("cam1", 4), ("cam2", 5)} in groups                       # same person: first visit, comeback, next camera
    assert {("cam1", 2), ("cam2", 6)} in groups
    assert {("cam1", 3)} in groups                                                  # overlaps track 1 on the same camera
    assert identities.of_track(r, "cam2", 5)["cameras"] == ["cam1", "cam2"]


def test_a_person_the_tracker_lost_for_a_second_gets_one_person_number(tmp_path):
    """Girl with a teddy: track 3 ends, track 5 starts 2 s later in the same place on the same camera. Different clothes in the data
    must not matter; a different person starting far away at the same moment must stay separate."""
    from webapp import identities
    rng = np.random.default_rng(5)
    common = rng.normal(size=64) * 3
    spec = {3: (0, 10, [100, 100, 140, 220]), 5: (12, 22, [110, 105, 150, 225]), 8: (12, 22, [600, 100, 640, 220])}
    obs, attrs, embs = [], [], {}
    for t, (t0, t1, box) in spec.items():
        obs += [{"camera_id": "cam1", "track_id": t, "frame": i, "timestamp": t0 + i, "class": "person", "confidence": .9, "bbox_xyxy": box} for i in range(t1 - t0)]
        attrs.append({"camera_id": "cam1", "track_id": t, "upper_color": "gray", "lower_color": "gray"})
        embs[str(t)] = {"embedding": list(map(float, rng.normal(size=64) + common))}      # unrelated appearances
    (tmp_path / "outputs").mkdir(); (tmp_path / "embeddings").mkdir()
    (tmp_path / "outputs" / "cam1_observations.json").write_text(json.dumps(obs))
    (tmp_path / "outputs" / "cam1_attributes.json").write_text(json.dumps(attrs))
    (tmp_path / "embeddings" / "cam1_embeddings.json").write_text(json.dumps({"camera_id": "cam1", "tracks": embs}))
    r = identities.build(tmp_path)
    groups = [{m["track_id"] for m in p["members"]} for p in r["people"]]
    assert {3, 5} in groups and {8} in groups
