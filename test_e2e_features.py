"""
End-to-end tests for clothing attributes + conversational Q&A + floor plan, through the real api.py.
Run:  pytest -q test_e2e_features.py      (or: python test_e2e_features.py)
Reuses the synthetic dataset built by test_e2e.py (never touches your real outputs/ folder).
"""
import json

import cv2
import numpy as np

from test_e2e import DATA, c      # builds the dataset, loads the real api.py

OUT = DATA / "outputs"


def _attr(cam, tid, up, lo, t0, t1, uc=0.9, lc=0.9):
    return {"camera_id": cam, "track_id": tid, "num_samples": 8, "first_seen_timestamp": t0, "last_seen_timestamp": t1,
            "upper_color": up, "upper_conf": uc, "lower_color": lo, "lower_conf": lc}


# cam2:4 matched to cam3:2 although clothing differs completely -> should be flagged as a conflict
(OUT / "cam1_attributes.json").write_text(json.dumps([_attr("cam1", 1, "red", "blue", 0, 10), _attr("cam1", 9, "white", "black", 50, 55)]))
(OUT / "cam2_attributes.json").write_text(json.dumps([_attr("cam2", 4, "red", "blue", 18, 30)]))
(OUT / "cam3_attributes.json").write_text(json.dumps([_attr("cam3", 2, "green", "black", 100, 110)]))
(DATA / "relationships").mkdir(exist_ok=True)
(DATA / "relationships" / "relationships.json").write_text(json.dumps([
    {"camera_id": "cam1", "track_a": 1, "track_b": 9, "start_timestamp": 3.0, "end_timestamp": 6.0,
     "duration_seconds": 3.0, "avg_proximity": 1.1}]))


def ask(q, **p):
    r = c.get("/qa", params={"question": q, **p})
    assert r.status_code == 200, r.json()
    return r.json()


# ================================================================= clothing attributes
def test_attributes_loaded_and_filtered():
    assert "routes_attributes" in c.get("/modules").json()["loaded"]
    assert c.get("/attributes").json()["count"] == 4
    assert c.get("/attributes", params={"upper_color": "red"}).json()["count"] == 2
    assert c.get("/attributes", params={"lower_color": "black", "camera_id": "cam3"}).json()["count"] == 1


def test_attribute_search():
    d = c.get("/attributes/search", params={"description": "person in a green jacket and black pants"}).json()
    assert d["understood_as"] == {"upper": "green", "lower": "black", "any": None}
    assert [(r["camera_id"], r["track_id"]) for r in d["results"]][0] == ("cam3", 2)
    assert c.get("/attributes/search", params={"description": "a tall man"}).status_code == 422


def test_verify_matches_flags_conflict_first():
    d = c.get("/attributes/verify_matches").json()
    assert d["counts"] == {"consistent": 1, "conflict": 1, "unknown": 0}
    first = d["matches"][0]
    assert (first["camera_a"], first["camera_b"], first["clothing"]["verdict"]) == ("cam2", "cam3", "conflict")


# ================================================================= conversational Q&A
def test_qa_where_did_person_go():
    r = ask("Where did the person in the red jacket go?")
    assert r["intent"] == "went_to" and "Route: cam1 → cam2 → cam3" in r["answer"]
    assert "unaccounted for 70.0s between cam2 and cam3" in r["answer"]
    assert "2 people match" in r["answer"]                                      # honest about ambiguity
    assert "camera 1 then camera 2" in r["speak"] and "→" not in r["speak"]
    assert " ," not in r["speak"] and ",," not in r["speak"] and ", ." not in r["speak"]       # clean text for speech


def test_qa_came_from():
    r = ask("Where did track 4 on cam2 come from?")
    assert r["intent"] == "came_from" and "came from cam1" in r["answer"]
    r = ask("where did track 2 come from")                                      # camera inferred: only cam3 has track 2
    assert "cam1 → cam2 → cam3" in r["answer"]


def test_qa_isolated_sighting_is_honest():
    r = ask("Where did track 9 on cam1 go?")
    assert "don't know where they went" in r["answer"]


def test_qa_journey_cameras_when_wearing():
    assert "Full journey: cam1 → cam2 → cam3" in ask("Show the journey of track 1 on cam1")["answer"]
    assert "3 cameras: cam1, cam2, cam3" in ask("Which cameras did track 1 on cam1 appear on?")["answer"]
    w = ask("When was track 4 on cam2 first seen?")["answer"]
    assert "First seen at 0.0s on cam1" in w and "last seen at 110.0s on cam3" in w
    assert "red top" in ask("What was track 4 on cam2 wearing?")["answer"]
    assert "green top" in ask("What is the person in the green jacket wearing?")["answer"]


def test_qa_with_whom_and_gaps():
    r = ask("Who was with track 1 on cam1?")
    assert "track 9" in r["answer"] and "white top" in r["answer"] and "3.0s together" in r["answer"]
    assert "was not seen together" in ask("Who was with track 4 on cam2?")["answer"]
    g = ask("Are there any blind spots in the journey of track 1 on cam1?")
    assert g["intent"] == "gaps" and "70.0s" in g["answer"]
    assert "No unexplained gaps" in ask("any gaps for track 1 on cam1", max_normal_gap=100)["answer"]


def test_qa_counts():
    r = ask("How many people were seen between 0 and 60 seconds?")
    assert r["data"] == {"sightings": 3, "people": 2}
    assert "3 sightings" in r["answer"] and "about 2 distinct people" in r["answer"]
    assert ask("how many people on cam3?")["data"]["sightings"] == 1
    assert "No one was seen" in ask("how many people on cam1 between 200 and 300 seconds")["answer"]
    assert ask("How many red cars were seen?")["data"]["vehicles"].__len__() == 2


def test_qa_camera_health_and_vehicles():
    assert "1 camera is down: cam2" in ask("Which cameras are down?")["answer"]
    v = ask("Where did the car MH12AB1234 go?")
    assert "last sighting" in v["answer"] and "cam3" in v["answer"] and "unaccounted for 83.0s" in v["answer"]
    assert "no sightings" in ask("Where did the car ZZ99ZZ9999 go?")["answer"]


def test_qa_never_invents():
    assert "can't find a sighting" in ask("Where did track 77 on cam1 go?")["answer"]
    assert "Nobody in the data matches" in ask("Where did the person in the purple jacket go?")["answer"]
    assert "Nobody in the data matches" in ask("show the journey of the person in a blue shirt and black pants")["answer"]
    assert "couldn't tell who" in ask("Where did he go?")["answer"]
    assert "didn't understand" in ask("what is the meaning of life")["answer"]
    assert c.get("/qa", params={}).status_code == 422
    assert len(c.get("/qa/examples").json()["examples"]) >= 5


# ================================================================= conversation: follow-up questions
def _ctx(r):
    ctx = r["context"]
    assert ctx, r
    if "plate" in ctx:
        return {"context_plate": ctx["plate"]}
    return {"context_camera": ctx["camera_id"], "context_track": ctx["track_id"]}


def test_qa_followups_use_the_previous_subject():
    first = ask("Where did the person in the green jacket go?")
    assert first["context"] == {"camera_id": "cam3", "track_id": 2} and first["used_previous_subject"] is False
    # no one is named any more: "he" must mean cam3 track 2
    w = ask("What was he wearing?", **_ctx(first))
    assert "cam3 track 2" in w["answer"] and "green top" in w["answer"] and w["used_previous_subject"] is True
    came = ask("And where did he come from?", **_ctx(w))
    assert came["used_previous_subject"] is True and "cam1 → cam2 → cam3" in came["answer"]
    assert came["context"] == {"camera_id": "cam3", "track_id": 2}                    # still remembered afterwards
    assert "70.0s" in ask("Any blind spots?", **_ctx(came))["answer"]
    assert "was not seen together" in ask("Who was he with?", **_ctx(came))["answer"]


def test_qa_followup_without_context_does_not_guess():
    r = ask("What was he wearing?")
    assert "couldn't tell who" in r["answer"] and r["used_previous_subject"] is False and r["context"] is None


def test_qa_new_subject_overrides_memory():
    first = ask("Where did track 4 on cam2 go?")
    r = ask("Where did track 1 on cam1 come from?", **_ctx(first))
    assert r["used_previous_subject"] is False and r["context"] == {"camera_id": "cam1", "track_id": 1}
    # an explicit clothing description with NO match must not silently fall back to the remembered person
    no = ask("Where did the person in the purple jacket go?", **_ctx(first))
    assert "Nobody in the data matches" in no["answer"] and no["used_previous_subject"] is False


def test_qa_context_survives_unrelated_questions_and_stale_context_is_ignored():
    first = ask("What was track 4 on cam2 wearing?")
    health = ask("Which cameras are down?", **_ctx(first))
    assert health["context"] == first["context"]                                     # not wiped by an unrelated question
    stale = ask("What was he wearing?", context_camera="cam9", context_track=1)       # no such sighting any more
    assert "couldn't tell who" in stale["answer"] and stale["context"] is None


def test_qa_bare_subject_means_find_them():
    r = ask("person in green jacket")                     # no question word at all
    assert r["intent"] == "journey" and "cam3" in r["answer"] and r["context"] == {"camera_id": "cam3", "track_id": 2}
    assert ask("track 4 on cam2")["intent"] == "journey"
    assert ask("MH12AB1234")["intent"] == "journey"
    assert "Nobody in the data matches" in ask("person in blue shirt")["answer"]      # honest when nobody matches
    assert "didn't understand" in ask("what is the meaning of life")["answer"]        # still refuses nonsense


def test_qa_vehicle_followup():
    first = ask("Where did the car MH12AB1234 go?")
    assert first["context"] == {"plate": "MH12AB1234"}
    r = ask("Where did it come from?", **_ctx(first))
    assert r["used_previous_subject"] is True and "first seen on cam1" in r["answer"]


# ================================================================= floor plan
import os  # noqa: E402

# cam1 / cam2 are calibrated (a simple scale: 640x360 px image <-> 20 x 12 m plan); cam3 has no calibration (pinned)
_SQUARE = {"image_points": [[0, 0], [640, 0], [640, 360], [0, 360]], "floor_points": [[0, 0], [20, 0], [20, 12], [0, 12]]}
(DATA / "floorplan.json").write_text(json.dumps({
    "width": 20, "height": 12, "scale": 40,
    "cameras": {"cam1": {"position": [10, 0.5], "calibration": _SQUARE},
                "cam2": {"position": [19.5, 6], "calibration": _SQUARE},
                "cam3": {"position": [2, 11]}}}))


def _walk(cam, tid, t0, t1, x0, y0, x1, y1, fps=10):
    n = int((t1 - t0) * 2) + 1
    out = []
    for i in range(n):
        f = i / (n - 1)
        cx, foot = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        out.append({"camera_id": cam, "track_id": tid, "frame": int((t0 + i * 0.5) * fps), "timestamp": round(t0 + i * 0.5, 2),
                    "class": "person", "confidence": 0.9, "bbox_xyxy": [int(cx - 15), int(foot - 80), int(cx + 15), int(foot)]})
    return out


# people now actually move (feet go from (100,120) to (500,280) px on cam1); timestamps/ids are unchanged
_f = DATA / "outputs"
(_f / "cam1_observations.json").write_text(json.dumps(_walk("cam1", 1, 0, 10, 100, 120, 500, 280) + _walk("cam1", 9, 50, 55, 300, 200, 320, 220)))
(_f / "cam2_observations.json").write_text(json.dumps(_walk("cam2", 4, 18, 30, 600, 300, 400, 150)))


def test_floorplan_config_reports_calibration():
    d = c.get("/floorplan/config").json()
    assert (d["width"], d["height"]) == (20, 12)
    assert d["cameras"]["cam1"]["calibrated"] is True and d["cameras"]["cam1"]["ok"] is True
    assert d["cameras"]["cam3"]["calibrated"] is False


def test_floorplan_journey_projects_feet_onto_the_plan():
    d = c.get("/floorplan/journey", params={"camera_id": "cam2", "track_id": 4}).json()
    assert [h["camera_id"] for h in d["hops"]] == ["cam1", "cam2", "cam3"]
    assert [h["mode"] for h in d["hops"]] == ["projected", "projected", "camera_position"]
    p = d["hops"][0]["points"]
    # feet (100,120) px -> (100*20/640, 120*12/360) = (3.125, 4.0) m ;  feet (500,280) px -> (15.625, 9.333) m
    assert abs(p[0]["x"] - 3.125) < 0.05 and abs(p[0]["y"] - 4.0) < 0.05
    assert abs(p[-1]["x"] - 15.625) < 0.05 and abs(p[-1]["y"] - 9.333) < 0.05
    assert d["hops"][2]["points"][0]["x"] == 2 and d["hops"][2]["points"][0]["y"] == 11    # pinned at cam3
    assert d["hops"][0]["out_of_bounds_ratio"] == 0.0
    a, b = d["connectors"]
    assert a["unexplained"] is False and 7 < a["gap_seconds"] < 9
    assert b["unexplained"] is True and b["gap_seconds"] == 70.0 and b["severity"] == "critical"
    assert len(d["gaps"]) == 1


def test_floorplan_render_is_a_real_picture_with_the_gap_drawn():
    r = c.get("/floorplan/render", params={"camera_id": "cam1", "track_id": 1})
    assert r.status_code == 200
    body = r.content if hasattr(r, "content") else r.json().content
    img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (480, 800, 3)
    red = (np.abs(img.astype(int) - np.array([50, 50, 230])).sum(axis=2) < 60).sum()     # the critical dashed connector
    assert red > 200, red
    assert (img.std(axis=(0, 1)) > 5).all()                                                # not a blank canvas


def test_floorplan_errors():
    assert c.get("/floorplan/journey", params={"camera_id": "cam9", "track_id": 1}).status_code == 404
    assert c.get("/floorplan/journey", params={"camera_id": "cam1"}).status_code == 422
    cfg = DATA / "floorplan.json"
    saved = cfg.read_text()
    try:
        cfg.write_text("{ not json")
        assert c.get("/floorplan/config").status_code == 500
        cfg.unlink()
        assert c.get("/floorplan/config").status_code == 404
    finally:
        cfg.write_text(saved)

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
