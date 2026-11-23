"""
Tests for the new web interface backend (routes_web.py + webapp/).

Run from your project folder (same as the other tests):
    python -m pytest -q test_e2e.py test_e2e_features.py test_web.py        # or:  python test_web.py

It builds its own synthetic cameras (webapp.demo_data) in a temp folder, points the web routes at it, and calls every
/web/*, /video, /stream endpoint through the real api.py. The processing test uses a stand-in for YOLO so it needs no
model download. Your real outputs/ folder is never touched.
"""
import json
import shutil
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from test_e2e import c                                  # the real api.py behind FastAPI's TestClient (also sets CATCHCAM_DATA_DIR)
import routes_web                                       # noqa: E402
from webapp import core, demo_data                      # noqa: E402

_DEMO = {}


def demo_dir() -> Path:
    if "dir" not in _DEMO:
        d = Path(tempfile.mkdtemp(prefix="catchcam_web_"))
        demo_data.build(d, webm=True)
        _DEMO["dir"] = d
    return _DEMO["dir"]


@contextmanager
def web(copy=False):
    """Point the web routes at the demo cameras. copy=True works on a private copy (for tests that add/replace files)."""
    d = demo_dir()
    if copy:
        d = Path(tempfile.mkdtemp(prefix="catchcam_webcopy_")) / "data"
        shutil.copytree(demo_dir(), d)
    old = routes_web.DATA_DIR
    routes_web.DATA_DIR = d
    try:
        yield d
    finally:
        routes_web.DATA_DIR = old


# ---------------------------------------------------------------- the interface files
def test_interface_is_served_and_cannot_escape_its_folder():
    r = c.get("/app/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "CatchCam" in r.text
    assert c.get("/app/js/main.js").status_code == 200 and c.get("/app/app.css").status_code == 200
    assert c.get("/app/nope.js").status_code == 404
    assert routes_web.home().status_code in (302, 307) and routes_web.home().headers["location"] == "/app/"
    try:
        routes_web.app_files("../api.py")
        assert False, "path traversal was allowed"
    except Exception as e:
        assert getattr(e, "status_code", None) == 403
    assert "web" in c.get("/modules").json()["loaded"][-1] or "routes_web" in c.get("/modules").json()["loaded"]


# ---------------------------------------------------------------- cameras and data
def test_cameras_overview():
    with web():
        cams = {x["camera_id"]: x for x in c.get("/web/cameras").json()["cameras"]}
    assert set(cams) == {"cam1", "cam2", "cam3"}
    x = cams["cam1"]
    assert x["has_video"] and x["plays_in_browser"] and x["num_tracks"] == 2 and x["status"] == "recorded"
    assert x["video_url"] == "/video/cam1" and abs(x["duration"] - 60) < 1


def test_find_by_clothing_gives_one_card_with_the_route_and_opens_it():
    with web():
        r = c.get("/web/find", params={"q": "person in green jacket"}).json()
    assert len(r["results"]) == 1, r["results"]                       # the same person on 3 cameras is ONE card
    card = r["results"][0]
    assert card["route"] == "cam1 → cam2 → cam3" and card["sightings"] == 3 and card["clothing"]["upper"] == "green"
    assert r["selected"]["camera_id"] in ("cam1", "cam2", "cam3") and "unaccounted" in r["answer"]


def test_find_follow_up_and_nonsense():
    with web():
        first = c.get("/web/find", params={"q": "person in green jacket"}).json()
        ctx = first["context"]
        w = c.get("/web/find", params={"q": "what was he wearing?", "context_camera": ctx["camera_id"], "context_track": ctx["track_id"]}).json()
        nothing = c.get("/web/find", params={"q": "person in purple jacket"}).json()
        gibberish = c.get("/web/find", params={"q": "xyzzy plugh"}).json()
        empty = c.get("/web/find", params={"q": "   "}).json()
    assert "green top" in w["answer"]
    assert nothing["results"] == [] and "Nobody in the data matches" in nothing["answer"] and nothing["selected"] is None
    assert gibberish["results"] == [] and "didn't understand" in gibberish["answer"]
    assert empty["answer"] is None and empty["examples"]


def test_subject_bundle_has_everything_the_screen_needs():
    with web():
        b = c.get("/web/subject", params={"camera_id": "cam2", "track_id": 1}).json()          # asking from the middle of the journey
        missing = c.get("/web/subject", params={"camera_id": "cam1", "track_id": 99})
    assert [h["camera_id"] for h in b["hops"]] == ["cam1", "cam2", "cam3"]
    assert [h["is_selected"] for h in b["hops"]] == [False, True, False]
    assert len(b["gaps"]) == 1 and abs(b["gaps"][0]["gap_seconds"] - 23.0) < 0.2 and b["gaps"][0]["severity"] == "suspicious" and b["has_blind_spot"]
    assert b["hops"][1]["clothing_check"]["verdict"] == "consistent" and b["hops"][1]["similarity"] == 0.91
    assert [(x["camera_id"], x["track_id"]) for x in b["companions"]] == [("cam1", 2)]
    assert b["floorplan_url"] is None and b["t_min"] < 3 and b["t_max"] > 58 and "unaccounted" in b["narrative"]
    assert missing.status_code == 404 and "No sighting" in missing.json()["detail"]
    with web():
        strict = c.get("/web/subject", params={"camera_id": "cam1", "track_id": 1, "critical_gap": 10}).json()
    assert strict["gaps"][0]["severity"] == "critical"


def test_overview_flags_the_journey_with_a_gap():
    with web():
        o = c.get("/web/overview").json()
    assert o["people_tracked"] == 6 and o["matches"] == 3 and o["with_clothing_data"] == 6 and o["cameras_down"] == []
    assert len(o["flagged_journeys"]) == 1 and o["flagged_journeys"][0]["route"] == "cam1 → cam2 → cam3"
    assert sum(sum(v) for v in o["activity"].values()) == 6 and o["match_conflicts"] == []


# ---------------------------------------------------------------- video and pictures
def test_video_supports_seeking():
    with web():
        full = c.get("/video/cam1")
        part = c.get("/video/cam1", headers={"Range": "bytes=0-99"})
        tail = c.get("/video/cam1", headers={"Range": "bytes=-50"})
        gone = c.get("/video/cam9")
    size = int(full.headers["content-length"])
    assert full.status_code == 200 and size > 10000 and full.headers["accept-ranges"] == "bytes"
    head, last = part.content, tail.content
    assert part.status_code == 206 and part.headers["content-range"] == f"bytes 0-99/{size}" and len(head) == 100
    assert tail.status_code == 206 and len(last) == 50 and head[:4] == b"\x1a\x45\xdf\xa3"       # WebM / Matroska header
    assert gone.status_code == 404


def test_range_parser_edge_cases():
    assert core.parse_range("bytes=10-", 100) == (10, 99) and core.parse_range("bytes=0-999", 100) == (0, 99)
    assert core.parse_range("bytes=-20", 100) == (80, 99) and core.parse_range(None, 100) is None
    assert core.parse_range("bytes=50-10", 100) is None and core.parse_range("bytes=abc-", 100) is None and core.parse_range("items=1-2", 100) is None


def test_frames_thumbs_and_boxes():
    with web():
        f = c.get("/web/frame", params={"camera_id": "cam1", "t": 5, "track_id": 1, "width": 320})
        thumb = c.get("/web/thumb", params={"camera_id": "cam1", "track_id": 1})
        nothing = c.get("/web/frame", params={"camera_id": "cam1", "t": 9999})
        no_thumb = c.get("/web/thumb", params={"camera_id": "cam1", "track_id": 77})
        boxes = c.get("/web/boxes", params={"camera_id": "cam1", "start": 0, "end": 20}).json()
        only = c.get("/web/boxes", params={"camera_id": "cam1", "start": 0, "end": 20, "track_id": 2}).json()
        too_long = c.get("/web/boxes", params={"camera_id": "cam1", "start": 0, "end": 5000})
    assert f.status_code == 200 and f.content[:2] == b"\xff\xd8" and f.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(f.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 320
    yellow = (np.abs(img.astype(int) - np.array([63, 210, 255])).sum(axis=2) < 60).sum()
    assert yellow > 20, "the followed person's box (yellow) is not drawn"
    assert thumb.status_code == 200 and thumb.content[:2] == b"\xff\xd8"
    assert nothing.status_code == 404 and no_thumb.status_code == 404
    assert boxes["rows"] and all(len(r) == 6 for r in boxes["rows"]) and {r[1] for r in boxes["rows"]} == {1, 2}
    assert only["rows"] and {r[1] for r in only["rows"]} == {2}
    assert too_long.status_code == 422


def test_server_drawn_stream_works_for_any_codec():
    with web():
        r = c.get("/stream/cam1", params={"start": 3, "end": 4, "speed": 8, "width": 320})
        gone = c.get("/stream/cam9")
    assert r.status_code == 200 and r.headers["content-type"].startswith("multipart/x-mixed-replace")
    body = r.content
    assert body.startswith(b"--frame") and body.count(b"--frame") >= 3 and b"\xff\xd8" in body
    assert gone.status_code == 404


# ---------------------------------------------------------------- adding footage
def test_upload_rules_and_safety():
    with web(copy=True) as d:
        video = (d / "videos" / "cam1.webm").read_bytes()
        bad_ext = c.post("/web/upload", params={"camera_id": "x", "filename": "notes.txt"}, content=b"hello")
        empty = c.post("/web/upload", params={"camera_id": "x", "filename": "a.mp4"}, content=b"")
        junk = c.post("/web/upload", params={"camera_id": "x", "filename": "a.mp4"}, content=b"this is not a video" * 100)
        bad_name = c.post("/web/upload", params={"camera_id": "!!!", "filename": "a.webm"}, content=video)
        ok = c.post("/web/upload", params={"camera_id": "lobby door", "filename": "x.webm"}, content=video)
        evil = c.post("/web/upload", params={"camera_id": "../../evil", "filename": "x.webm"}, content=video)
        listed = {x["camera_id"] for x in c.get("/web/cameras").json()["cameras"]}
        stored = sorted(p.name for p in (d / "videos").iterdir())
        outside = [p for p in d.parent.iterdir() if p.name != "data"]
    assert bad_ext.status_code == 422 and "Can't use" in bad_ext.json()["detail"]
    assert empty.status_code == 422 and junk.status_code == 422 and "can't be read" in junk.json()["detail"]
    assert bad_name.status_code == 422
    assert ok.status_code == 200 and ok.json()["camera_id"] == "lobby_door" and ok.json()["plays_in_browser"] is True
    assert evil.json()["camera_id"] == "evil" and outside == []                      # sanitised, and nothing written outside the data folder
    assert {"lobby_door", "evil"} <= listed and "x.part" not in " ".join(stored) and not any(s.endswith(".part") for s in stored)


class _Tensor:
    def __init__(self, a): self.a = np.array(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def int(self): return self
    def tolist(self): return self.a.tolist()


@contextmanager
def fake_yolo():
    fake_yolo.calls = []
    """Stands in for ultralytics.YOLO: 'tracks' one person walking across every frame."""
    class _Boxes:
        def __init__(self, i): self.xyxy, self.id, self.conf = _Tensor([[10 + i, 40, 60 + i, 200]]), _Tensor([7]), _Tensor([0.9])

    class _Result:
        def __init__(self, i): self.boxes = _Boxes(i)

    class YOLO:
        def __init__(self, weights): self.weights = weights
        def track(self, source, stream=True, **kw):
            fake_yolo.calls.append(kw)
            n = int(cv2.VideoCapture(source).get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(0, n, kw.get("vid_stride", 1)):                # like ultralytics: frames in between are never detected
                res = _Result(i)
                if getattr(fake_yolo, "extra_ghosts", False) and i == 2:                 # one frame with a second, low-confidence "person"
                    res.boxes.xyxy, res.boxes.id, res.boxes.conf = _Tensor([[10 + i, 40, 60 + i, 200], [300, 20, 330, 90]]), _Tensor([7, 99]), _Tensor([0.9, 0.4])
                yield res

    mod = types.ModuleType("ultralytics")
    mod.YOLO = YOLO
    old = sys.modules.get("ultralytics")
    sys.modules["ultralytics"] = mod
    try:
        yield
    finally:
        if old is None: sys.modules.pop("ultralytics", None)
        else: sys.modules["ultralytics"] = old


def _wait_job(job_id, seconds=90):
    end = time.time() + seconds
    while time.time() < end:
        j = c.get(f"/web/jobs/{job_id}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.3)
    raise AssertionError("job did not finish")


@contextmanager
def embedding_models(kind):
    """The appearance-matching step uses YOUR extract_embeddings.py / match_cameras.py (torch, a downloaded model). Whether that works
    must not decide this test, so stand in for them: kind='missing' (not installed) or 'broken' (raises, e.g. no internet for the weights)."""
    names = ("extract_embeddings", "match_cameras")
    old = {n: sys.modules.get(n) for n in names}
    try:
        if kind == "missing":
            for n in names:
                sys.modules[n] = None
        else:
            def boom(*a, **k): raise RuntimeError("could not download the model weights")
            for n in names:
                m = types.ModuleType(n)
                m.extract_embeddings = boom
                m.match_cameras = boom
                sys.modules[n] = m
        yield
    finally:
        for n, v in old.items():
            if v is None: sys.modules.pop(n, None)
            else: sys.modules[n] = v


def test_processing_pipeline_runs_and_reports_progress():
    with web(copy=True) as d, fake_yolo(), embedding_models("missing"):
        video = (d / "videos" / "cam1.webm").read_bytes()
        assert c.post("/web/upload", params={"camera_id": "lab", "filename": "lab.webm"}, content=video).status_code == 200
        job = c.post("/web/process", params={"camera_id": "lab", "time_offset": 5, "skip_empty": 0}).json()
        assert job["state"] in ("queued", "running", "done") and job["camera_id"] == "lab"
        done = _wait_job(job["id"])
        obs = json.loads((d / "outputs" / "lab_observations.json").read_text())
        cams = {x["camera_id"]: x for x in c.get("/web/cameras").json()["cameras"]}
        offsets = json.loads((d / "time_offsets.json").read_text())
        again = c.post("/web/process", params={"camera_id": "nobody"})
    assert done["state"] == "done" and done["percent"] == 100, done
    assert any("Found 1 people" in n for n in done["notes"]) and any("Skipped appearance matching" in n for n in done["notes"])
    assert obs and obs[0]["camera_id"] == "lab" and obs[0]["track_id"] == 7 and abs(obs[0]["timestamp"] - 5.0) < 0.01     # shared clock = offset + frame/fps
    assert all(isinstance(v, int) for o in obs for v in o["bbox_xyxy"]), obs[0]["bbox_xyxy"]   # whole pixels: extract_embeddings.py slices frames with them
    assert {"frame", "class", "confidence", "bbox_xyxy"} <= set(obs[0])
    assert cams["lab"]["num_tracks"] == 1 and cams["lab"]["offset"] == 5.0 and offsets["lab"] == 5.0
    assert again.status_code == 404


def test_a_failing_appearance_step_keeps_the_tracking_and_says_what_failed():
    with web(copy=True) as d, fake_yolo(), embedding_models("broken"):
        video = (d / "videos" / "cam1.webm").read_bytes()
        c.post("/web/upload", params={"camera_id": "lab3", "filename": "lab3.webm"}, content=video)
        done = _wait_job(c.post("/web/process", params={"camera_id": "lab3"}).json()["id"])
        tracked = (d / "outputs" / "lab3_observations.json").exists()
    assert done["state"] == "done" and tracked, done                       # the tracking work is kept, the job does not end in an error
    assert any("Appearance matching failed" in n and "model weights" in n for n in done["notes"]), done["notes"]


def test_processing_failure_is_reported_not_swallowed():
    with web(copy=True) as d:
        video = (d / "videos" / "cam1.webm").read_bytes()
        c.post("/web/upload", params={"camera_id": "lab2", "filename": "lab2.webm"}, content=video)
        old = sys.modules.pop("ultralytics", None)
        try:
            sys.modules["ultralytics"] = None                      # makes `import ultralytics` fail, like a machine without it
            job = c.post("/web/process", params={"camera_id": "lab2"}).json()
            done = _wait_job(job["id"])
        finally:
            sys.modules.pop("ultralytics", None)
            if old is not None: sys.modules["ultralytics"] = old
    assert done["state"] == "error" and done["error"] and "ultralytics" in done["error"]


def test_second_job_for_the_same_camera_is_refused_while_one_runs():
    from webapp import jobs
    mgr = jobs.JobManager()
    import threading
    gate = threading.Event()
    j = mgr.start("process", "camX", lambda job: gate.wait(5))
    try:
        mgr.start("process", "camX", lambda job: None)
        assert False, "second job should have been refused"
    except RuntimeError as e:
        assert "already being processed" in str(e)
    assert mgr.start("process", "camY", lambda job: None)               # a different camera is fine
    gate.set()
    for _ in range(50):
        if j.state == "done": break
        time.sleep(0.1)
    assert j.state == "done"


def test_live_stream_registration_validates_the_address():
    with web(copy=True) as d:
        bad_name = c.post("/web/streams", params={"camera_id": "!!!", "url": "0"})
        unreachable = c.post("/web/streams", params={"camera_id": "door", "url": "99"})              # no such webcam
        registered = "door" in json.loads((d / "streams.json").read_text()) if (d / "streams.json").exists() else False
    assert bad_name.status_code == 422 and unreachable.status_code == 422 and "Couldn't open" in unreachable.json()["detail"]
    assert registered is False


def test_formats_a_browser_cannot_play_are_detected_and_converted():
    with web(copy=True) as d:
        p = d / "videos" / "old.avi"
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 8, (320, 180))
        for i in range(40):
            w.write(np.full((180, 320, 3), i * 5, np.uint8))
        w.release()
        core.register_video(d, "old", p)
        before = {x["camera_id"]: x for x in c.get("/web/cameras").json()["cameras"]}["old"]
        assert before["plays_in_browser"] is False and before["stream_url"] == "/stream/old"          # falls back to the server stream
        if shutil.which("ffmpeg"):
            assert before["can_convert"] is True
            done = _wait_job(c.post("/web/convert", params={"camera_id": "old"}).json()["id"])
            after = {x["camera_id"]: x for x in c.get("/web/cameras").json()["cameras"]}["old"]
            assert done["state"] == "done" and after["plays_in_browser"] is True and (d / "videos" / "old.web.mp4").exists()
        else:
            assert before["can_convert"] is False


def test_time_offset_shifts_the_video_onto_the_shared_clock():
    with web(copy=True) as d:
        (d / "time_offsets.json").write_text(json.dumps({"cam2": 100}))
        cam2 = {x["camera_id"]: x for x in c.get("/web/cameras").json()["cameras"]}["cam2"]
        # the observations already sit on the shared clock (18s-30s), so a frame at t=120 is second 20 of the file
        shifted = c.get("/web/frame", params={"camera_id": "cam2", "t": 105})
        outside = c.get("/web/frame", params={"camera_id": "cam2", "t": 50})                   # before the file starts
    assert cam2["offset"] == 100 and shifted.status_code == 200 and outside.status_code == 404


def test_delete_camera_removes_its_footage_results_and_matches():
    with web(copy=True) as d:
        files = [d / "videos" / "cam2.webm", d / "outputs" / "cam2_observations.json"]
        assert all(f.exists() for f in files)
        before = json.loads((d / "matches" / "cross_camera_matches.json").read_text())
        assert any("cam2" in (m["camera_a"], m["camera_b"]) for m in before)
        r = c.delete("/web/cameras/cam2")
        assert r.status_code == 200 and r.json()["camera_id"] == "cam2"
        assert not any(f.exists() for f in files)
        ids = [x["camera_id"] for x in c.get("/web/cameras").json()["cameras"]]
        assert "cam2" not in ids and {"cam1", "cam3"} <= set(ids)               # the others are untouched
        after = json.loads((d / "matches" / "cross_camera_matches.json").read_text())
        assert after == []                                                       # every demo match involved cam2
        # deleting only drops the matches that mention the deleted camera
        (d / "matches" / "cross_camera_matches.json").write_text(json.dumps(
            [dict(before[0], camera_a="cam1", camera_b="cam3"), dict(before[0], camera_a="cam1", camera_b="camZ")]))
        assert c.delete("/web/cameras/cam3").status_code == 200
        kept = json.loads((d / "matches" / "cross_camera_matches.json").read_text())
        assert [m["camera_b"] for m in kept] == ["camZ"]
        assert not (d / "videos" / "cam3.webm").exists() and (d / "videos" / "cam1.webm").exists()
        assert c.delete("/web/cameras/cam2").status_code == 404                  # already gone
        assert c.get("/web/frame", params={"camera_id": "cam2", "t": 20}).status_code == 404
        # the same name can be used again for new footage
        up = c.post("/web/upload", params={"camera_id": "cam2", "filename": "x.webm"}, content=(d / "videos" / "cam1.webm").read_bytes())
        assert up.status_code == 200


def test_delete_camera_is_refused_while_it_is_processing_and_rejects_bad_names():
    from webapp import jobs
    import threading
    gate = threading.Event()
    with web(copy=True) as d:
        j = jobs.MANAGER.start("process", "cam3", lambda job: gate.wait(5))
        try:
            r = c.delete("/web/cameras/cam3")
            assert r.status_code == 409 and (d / "videos" / "cam3.webm").exists()
        finally:
            gate.set()
            for _ in range(50):
                if j.state == "done": break
                time.sleep(0.1)
        assert c.delete("/web/cameras/..").status_code in (404, 422)
        assert c.delete("/web/cameras/nope").status_code == 404


def test_speed_and_sensitivity_options_reach_the_detector():
    with web(copy=True) as d, fake_yolo(), embedding_models("missing"):
        video = (d / "videos" / "cam1.webm").read_bytes()
        c.post("/web/upload", params={"camera_id": "lab", "filename": "lab.webm"}, content=video)
        normal = c.post("/web/process", params={"camera_id": "lab", "skip_empty": 0}).json()
        _wait_job(normal["id"])
        n_normal = len(json.loads((d / "outputs" / "lab_observations.json").read_text()))
        fast = c.post("/web/process", params={"camera_id": "lab", "speed": "fastest", "sensitivity": "high", "skip_empty": 0}).json()
        _wait_job(fast["id"])
        obs = json.loads((d / "outputs" / "lab_observations.json").read_text())
        bad = c.post("/web/process", params={"camera_id": "lab", "speed": "warp"})
        calls = list(fake_yolo.calls)
    assert calls[0]["vid_stride"] == 1 and calls[0]["conf"] == 0.35 and "imgsz" not in calls[0]
    assert calls[1]["vid_stride"] == 5 and calls[1]["conf"] == 0.2 and calls[1]["imgsz"] == 480
    assert 0 < len(obs) < n_normal                                          # fewer frames were looked at
    assert all(o["frame"] % 5 == 0 for o in obs)                             # frame numbers stay true to the video, so the clock is right
    assert bad.status_code == 422


def test_ghost_tracks_are_removed_and_real_ones_kept():
    from webapp import cleaning

    def track(tid, n, conf, box=(100, 100, 160, 300), step=3, dx=0, cam="c"):
        return [{"camera_id": cam, "track_id": tid, "frame": i * step, "timestamp": i * step / 25, "class": "person", "confidence": conf,
                 "bbox_xyxy": [box[0] + dx * i, box[1], box[2] + dx * i, box[3]]} for i in range(n)]
    rows = (track(1, 60, .85, dx=8)                    # a person walking across: keep
            + track(2, 400, .91)                       # someone sitting at a desk for a long time: keep (not moving, but long and confident)
            + track(3, 1, .40) + track(4, 2, .39)      # blips: 1-2 detections
            + track(5, 15, .44)                        # low confidence and under ~2 s (15 detections x every 3rd frame = 45 frames)
            + track(6, 7, .55)                         # 7 detections x every 3rd frame = ~0.8 s, confident, still: a flicker on one spot
            + track(7, 9, .8, dx=15))                  # short but confident and moving: keep
    assert cleaning.ghost_tracks(rows) == {3, 4, 5, 6}, cleaning.ghost_tracks(rows)
    # counting is in video frames: the same people saved every frame instead of every 3rd give the same answer
    dense = (track(1, 180, .85, dx=3, step=1) + track(3, 3, .40, step=1) + track(5, 45, .44, step=1) + track(2, 1200, .9, step=1))
    assert cleaning.ghost_tracks(dense) == {3, 5}


def test_cleaning_a_camera_updates_every_file_that_mentions_the_ghosts():
    with web(copy=True) as d:
        obs = json.loads((d / "outputs" / "cam1_observations.json").read_text())
        ghost = max(r["track_id"] for r in obs) + 50
        obs += [{"camera_id": "cam1", "track_id": ghost, "frame": 4, "timestamp": 0.4, "class": "person", "confidence": .37, "bbox_xyxy": [5, 5, 40, 90]}]
        (d / "outputs" / "cam1_observations.json").write_text(json.dumps(obs))
        attrs = json.loads((d / "outputs" / "cam1_attributes.json").read_text())
        (d / "outputs" / "cam1_attributes.json").write_text(json.dumps(attrs + [dict(attrs[0], track_id=ghost)]))
        m = json.loads((d / "matches" / "cross_camera_matches.json").read_text())
        (d / "matches" / "cross_camera_matches.json").write_text(json.dumps(m + [dict(m[0], track_a=ghost)]))
        n_tracks = lambda: {x["camera_id"]: x["num_tracks"] for x in c.get("/web/cameras").json()["cameras"]}["cam1"]
        before = n_tracks()
        r = c.post("/web/clean", params={"camera_id": "cam1"})
        assert r.status_code == 200 and r.json()["removed"] == 1, r.text
        assert n_tracks() == before - 1
        assert all(a["track_id"] != ghost for a in json.loads((d / "outputs" / "cam1_attributes.json").read_text()))
        assert all(x["track_a"] != ghost for x in json.loads((d / "matches" / "cross_camera_matches.json").read_text()))
        assert (d / "outputs" / "cam1_observations.before_cleanup.json").exists()            # nothing is lost for good
        assert c.post("/web/clean", params={"camera_id": "cam1"}).json()["removed"] == 0     # nothing more to remove
        assert c.post("/web/clean").json()["removed"] == 0                                    # all cameras
        assert c.post("/web/clean", params={"camera_id": "nope"}).status_code == 404


def test_processing_reports_and_removes_false_detections():
    with web(copy=True) as d, fake_yolo(), embedding_models("missing"):
        fake_yolo.extra_ghosts = True
        try:
            video = (d / "videos" / "cam1.webm").read_bytes()
            c.post("/web/upload", params={"camera_id": "lab", "filename": "lab.webm"}, content=video)
            done = _wait_job(c.post("/web/process", params={"camera_id": "lab", "skip_empty": 0}).json()["id"])
            obs = json.loads((d / "outputs" / "lab_observations.json").read_text())
        finally:
            fake_yolo.extra_ghosts = False
    assert done["state"] == "done"
    assert {o["track_id"] for o in obs} == {7}                                  # the blip (track 99) is gone, the walker stays
    assert any("Ignored 1 false detections" in n for n in done["notes"]), done["notes"]


def _lobby_video(path, walkers=((30, 50), (200, 220)), frames=300, busy=False):
    """A static scene with a person-sized block that walks across during the given frame ranges (busy=True: someone is always there)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for f in range(frames):
        img = np.full((240, 320, 3), 200, np.uint8)
        img[:, :30] = 120
        for a, b in ([(0, frames)] if busy else walkers):
            if a <= f < b:
                x = 40 + (f - a) * 8 % 220
                img[60:200, x:x + 40] = (40, 40, 40)
        w.write(img)
    w.release()


def test_empty_stretches_are_skipped_and_times_stay_true_to_the_video():
    from webapp import tracker
    tmp = Path(tempfile.mkdtemp())
    _lobby_video(tmp / "quiet.avi")
    with fake_yolo():
        notes = []
        out = tracker.run(str(tmp / "quiet.avi"), "lobby", str(tmp / "o"), skip_empty=True, notes=notes, time_offset=100)
        rows = json.loads(Path(out).read_text())
        analysed = len(rows)
        full = json.loads(Path(tracker.run(str(tmp / "quiet.avi"), "lobby", str(tmp / "o2"), skip_empty=False)).read_text())
    assert 0 < analysed < len(full) * 0.5, (analysed, len(full))                      # far fewer frames were analysed
    assert sorted({r["track_id"] for r in rows}) == [1, 2]                              # two separate visits -> two people, numbered 1, 2
    # frames inside each visit, converted back to the video's own frame numbers (and the offset added)
    first = [r for r in rows if r["track_id"] == 1]
    second = [r for r in rows if r["track_id"] == 2]
    assert all(15 <= r["frame"] <= 65 for r in first) and all(185 <= r["frame"] <= 235 for r in second), (first[0]["frame"], second[0]["frame"])
    assert all(abs(r["timestamp"] - (100 + r["frame"] / 10)) < 0.01 for r in rows)
    assert any("Skipped the empty stretches" in n for n in notes), notes


def test_a_busy_camera_is_analysed_in_full_and_a_dead_one_says_so():
    from webapp import tracker
    tmp = Path(tempfile.mkdtemp())
    _lobby_video(tmp / "busy.avi", busy=True)
    _lobby_video(tmp / "dead.avi", walkers=())
    with fake_yolo():
        notes = []
        rows = json.loads(Path(tracker.run(str(tmp / "busy.avi"), "b", str(tmp / "o"), skip_empty=True, notes=notes)).read_text())
        assert len(rows) == 300 and {r["track_id"] for r in rows} == {7} and not notes          # nothing to skip: the usual single pass
        notes = []
        rows = json.loads(Path(tracker.run(str(tmp / "dead.avi"), "d", str(tmp / "o"), skip_empty=True, notes=notes)).read_text())
    assert rows == [] and any("No movement found" in n for n in notes), notes


if __name__ == "__main__":
    import traceback
    ok = bad = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); ok += 1; print("PASS", name)
            except Exception:
                bad += 1; print("FAIL", name); traceback.print_exc()
    print(f"{ok} passed, {bad} failed")
    raise SystemExit(1 if bad else 0)
