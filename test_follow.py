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
