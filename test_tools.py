"""
End-to-end tests for the accuracy check, the PDF incident report and the watchlist alerts, through the real api.py.

    python -m pytest -q test_e2e.py test_e2e_features.py test_web.py test_tools.py        # or:  python test_tools.py

Uses the same synthetic cameras as test_web.py (built once in a temp folder). Your own data is never touched.
"""
import json
import sys
from pathlib import Path

from test_e2e import c                                   # the real api.py behind FastAPI's TestClient
from test_web import web                                  # noqa: E402  (points the web routes at the demo cameras)
import routes_alerts, routes_eval, routes_report         # noqa: E402,E401

TRUTH = [("cam1", 1, "Asha"), ("cam2", 1, "Asha"), ("cam3", 1, "Asha"), ("cam1", 2, "Ravi"), ("cam2", 2, "Ravi"), ("cam3", 2, "Meena")]
SQUARE = {"image_points": [[0, 0], [640, 0], [640, 360], [0, 360]], "floor_points": [[0, 0], [20, 0], [20, 12], [0, 12]]}


class tools:
    """routes_eval / routes_report / routes_alerts each read their own DATA_DIR: point all of them at the same folder"""
    def __init__(self, copy=True):
        self.cm = web(copy=copy)

    def __enter__(self):
        self.d = self.cm.__enter__()
        self.old = [(m, m.DATA_DIR) for m in (routes_eval, routes_report, routes_alerts)]
        for m, _ in self.old:
            m.DATA_DIR = self.d
        return self.d

    def __exit__(self, *a):
        for m, old in self.old:
            m.DATA_DIR = old
        return self.cm.__exit__(*a)


def post_json(path, obj):
    return c.post(path, content=json.dumps(obj))


# ---------------------------------------------------------------- they are wired in without touching api.py
def test_new_screens_and_routes_are_reachable():
    assert c.get("/app/js/accuracy.js").status_code == 200 and c.get("/app/js/alerts.js").status_code == 200
    # Ask for each address instead of reading FastAPI's internal route list (its shape differs between versions).
    # A route that is not there answers {"detail": "Not Found"}; one that is there answers something else (data, or 422 for a missing parameter).
    for p in ("/eval/tracks", "/eval/run", "/report/journey.pdf", "/alerts", "/alerts/count"):
        r = c.get(p)
        assert not (r.status_code == 404 and r.json() == {"detail": "Not Found"}), f"{p} is not wired in"


# ---------------------------------------------------------------- accuracy check
def test_accuracy_check_end_to_end():
    with tools():
        t = c.get("/eval/tracks").json()
        assert len(t["tracks"]) == 6 and t["labelled"] == 0 and t["tracks"][0]["thumb_url"].startswith("/web/thumb")
        assert c.get("/eval/run").json()["status"] == "need_labels"
        assert post_json("/eval/labels", {"labels": [{"camera_id": a, "track_id": b, "person": p} for a, b, p in TRUTH]}).json() == {"saved": 6}
        r = c.get("/eval/run").json()
        assert r["status"] == "ok" and r["precision"] == 1.0 and r["recall"] == 1.0 and r["journeys"]["fully_correct"] == 2
        assert c.get("/eval/tracks").json()["labelled"] == 6
        md = c.get("/eval/report.md")
        assert md.status_code == 200 and "attachment" in md.headers["content-disposition"] and b"Precision | 100.0%" in md.content
        # CatchCam is wrong if I say Ravi's cam2 track is actually Meena
        wrong = [{"camera_id": a, "track_id": b, "person": ("Meena" if (a, b) == ("cam2", 2) else p)} for a, b, p in TRUTH]
        post_json("/eval/labels", {"labels": wrong})
        r = c.get("/eval/run").json()
        assert r["precision"] < 1.0 and r["pairs"]["false_matches"] >= 1 and r["false_match_list"]


def test_accuracy_labels_are_validated():
    with tools():
        assert c.post("/eval/labels", content=b"not json").status_code == 422
        assert c.post("/eval/labels", content=json.dumps({"nope": 1})).status_code == 422
        r = post_json("/eval/labels", {"labels": [{"camera_id": "cam9", "track_id": 1, "person": "X"}]})
        assert r.status_code == 422 and "does not exist" in r.json()["detail"]
        r = post_json("/eval/labels", {"labels": [{"camera_id": "cam1", "track_id": 1, "person": "<script>"}]})
        assert r.status_code == 422 and "not a usable person name" in r.json()["detail"]


# ---------------------------------------------------------------- PDF report
def test_report_is_a_real_pdf_with_pictures_from_each_camera():
    with tools():
        r = c.get("/report/journey.pdf", params={"camera_id": "cam1", "track_id": 1})
        assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
        assert r.content[:5] == b"%PDF-" and r.headers["content-disposition"].endswith('catchcam_report_cam1_track1.pdf"')
        assert r.content.count(b"/Subtype /Image") >= 3                                      # a snapshot for each of the three cameras
        try:
            import pypdf, io
            text = " ".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(r.content)).pages)
            for s in ("CatchCam incident report", "cam1", "23s unaccounted", "appearance match 91%", "How to read this report"):
                assert s in text, s
        except ImportError:
            pass


def test_report_includes_the_floor_plan_when_one_is_set_up():
    with tools() as d:
        plain = len(c.get("/report/journey.pdf", params={"camera_id": "cam1", "track_id": 1}).content)
        (d / "floorplan.json").write_text(json.dumps({"width": 20, "height": 12, "scale": 40, "cameras": {
            "cam1": {"position": [10, 0.5], "calibration": SQUARE}, "cam2": {"position": [19.5, 6], "calibration": SQUARE}, "cam3": {"position": [2, 11]}}}))
        with_plan = c.get("/report/journey.pdf", params={"camera_id": "cam1", "track_id": 1}).content
        assert len(with_plan) > plain + 3000 and with_plan.count(b"/Subtype /Image") >= 4
        (d / "floorplan.json").write_text("{broken")                                          # a broken floor plan must not break the report
        assert c.get("/report/journey.pdf", params={"camera_id": "cam1", "track_id": 1}).content[:5] == b"%PDF-"


def test_report_without_footage_still_works_and_unknown_people_are_404():
    with tools() as d:
        for f in (d / "videos").glob("*"):
            f.unlink()
        r = c.get("/report/journey.pdf", params={"camera_id": "cam2", "track_id": 1})
        assert r.status_code == 200 and r.content[:5] == b"%PDF-" and r.content.count(b"/Subtype /Image") == 0
        r = c.get("/report/journey.pdf", params={"camera_id": "cam1", "track_id": 99})
        assert r.status_code == 404 and "No sighting" in r.json()["detail"]


def test_report_says_how_to_fix_a_missing_package():
    with tools():
        # earlier tests already imported reportlab (and its submodules), so forget all of it first
        names = [k for k in sys.modules if k == "reportlab" or k.startswith("reportlab.") or k == "reports.core"]
        saved = {k: sys.modules.pop(k) for k in names}
        sys.modules["reportlab"] = None                                                        # as if it was never installed
        try:
            r = c.get("/report/journey.pdf", params={"camera_id": "cam1", "track_id": 1})
            assert r.status_code == 501 and "pip install reportlab" in r.json()["detail"]
        finally:
            sys.modules.pop("reportlab", None)
            sys.modules.update(saved)


# ---------------------------------------------------------------- alerts
def test_alerts_end_to_end():
    with tools() as d:
        assert c.get("/alerts").json() == {"alerts": [], "unacknowledged": 0, "items": [], "notifications": {"webhook": False, "telegram": False}}
        r = c.post("/alerts/watch", params={"kind": "clothing", "value": "red jacket"})
        assert r.status_code == 200 and r.json()["existing_hits"] == 2                          # Ravi on cam1 and on cam2
        a = c.get("/alerts").json()
        assert a["unacknowledged"] == 2 and all(x["historical"] for x in a["alerts"]) and a["alerts"][0]["link"].startswith("#investigate/cam")
        assert c.get("/alerts/count").json() == {"unacknowledged": 2}
        assert c.post("/alerts/ack", params={"alert_id": a["alerts"][0]["id"]}).json() == {"acknowledged": 1}
        assert c.post("/alerts/ack", params={"everything": "true"}).json() == {"acknowledged": 1}
        assert c.get("/alerts/count").json() == {"unacknowledged": 0}
        # new footage: a second red jacket appears on cam3
        f = d / "outputs" / "cam3_attributes.json"
        rows = json.loads(f.read_text())
        f.write_text(json.dumps(rows + [{**rows[0], "track_id": 9, "upper_color": "red", "upper_conf": 1.0}]))
        n = c.get("/alerts").json()
        assert n["unacknowledged"] == 1 and n["alerts"][0]["historical"] is False and n["alerts"][0]["camera_id"] == "cam3"
        item = n["items"][0]["id"]
        assert c.delete(f"/alerts/watch/{item}").json() == {"removed": item}
        assert c.delete(f"/alerts/watch/{item}").status_code == 404


def test_alert_inputs_are_validated():
    with tools():
        for params, text in [({"kind": "plate", "value": "x"}, "4 to 11"), ({"kind": "clothing", "value": "nice person"}, "colour"),
                             ({"kind": "bogus"}, "kind must be"), ({"kind": "gap", "value": "99999"}, "between 1 and 3600")]:
            r = c.post("/alerts/watch", params=params)
            assert r.status_code == 422 and text in r.json()["detail"], (params, r.json())
        assert c.post("/alerts/ack").status_code == 422
        # the demo cameras have a 23 s gap (cam2 -> cam3) and a 10.9 s gap (cam1 -> cam2)
        assert c.post("/alerts/watch", params={"kind": "gap", "value": "15"}).json()["existing_hits"] == 1
        assert c.post("/alerts/watch", params={"kind": "gap", "value": "10"}).json()["existing_hits"] == 2


def test_notification_settings_cannot_be_changed_through_the_api():
    with tools():
        for p in ("/alerts/config", "/alerts/webhook", "/alerts/notify"):
            assert c.post(p, params={"url": "http://evil.example"}).status_code == 404


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
