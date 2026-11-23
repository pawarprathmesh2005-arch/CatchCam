"""Tests for watchlist alerts (no video needed). Run:  python -m pytest -q alerts/test_alerts.py   or   python alerts/test_alerts.py"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from alerts import core as al, notify  # noqa: E402
import test_evaluation as te  # noqa: E402  (its synthetic cameras: Asha cam1#1 > cam2#1 > cam3#1 with a 22 s gap before cam3)

for _k in ("CATCHCAM_WEBHOOK_URL", "CATCHCAM_TELEGRAM_TOKEN", "CATCHCAM_TELEGRAM_CHAT"):
    os.environ.pop(_k, None)


def attrs(cam, tid, up, lo, a, b):
    return {"camera_id": cam, "track_id": tid, "num_samples": 5, "first_seen_timestamp": a, "last_seen_timestamp": b,
            "upper_color": up, "upper_conf": 1.0, "lower_color": lo, "lower_conf": 1.0}


def veh(cam, tid, plate, a=5.0, b=9.0):
    return {"camera_id": cam, "track_id": tid, "vehicle_type": "car", "color": "red", "plate": plate, "first_seen_timestamp": a, "last_seen_timestamp": b}


def make():
    d = te.make(labels=None)
    out = d / "outputs"
    (out / "cam1_attributes.json").write_text(json.dumps([attrs("cam1", 1, "green", "black", 2, 14), attrs("cam1", 2, "red", "blue", 4, 16)]))
    (out / "cam2_attributes.json").write_text(json.dumps([attrs("cam2", 1, "green", "black", 18, 30)]))
    (out / "cam1_vehicles.json").write_text(json.dumps([veh("cam1", 7, "MH12AB1234")]))
    return d


def grow(d, name, row):
    """a new sighting appears in the data (as if new footage was processed)"""
    f = d / "outputs" / name
    rows = json.loads(f.read_text()) if f.exists() else []
    f.write_text(json.dumps(rows + [row]))


class Catcher:
    """a throw-away web server standing in for Slack / Telegram"""
    def __init__(self, status=200):
        self.got = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                outer.got.append((self.path, json.loads(self.rfile.read(int(self.headers["content-length"])))))
                self.send_response(status); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()


def test_watchlist_validation():
    d = make()
    for kind, value, text in [("plate", "AB", "4 to 11"), ("clothing", "a nice person", "colour"), ("gap", "soon", "number of seconds"),
                              ("gap", "0", "between 1 and 3600"), ("nonsense", "x", "kind must be"), ("plate", "x" * 90, "too long")]:
        try:
            al.add_item(d, kind, value)
        except ValueError as e:
            assert text in str(e), (kind, value, e)
        else:
            raise AssertionError(f"should refuse {kind} {value!r}")
    it = al.add_item(d, "plate", "mh12 ab-1234")
    assert it["value"] == "MH12AB1234" and it["label"] == "Plate MH12AB1234" and it["existing_hits"] == 1
    try:
        al.add_item(d, "plate", "MH12AB1234")
    except ValueError as e:
        assert "already on the watchlist" in str(e)
    else:
        raise AssertionError("duplicate accepted")


def test_what_was_already_in_the_footage_is_shown_but_marked():
    d = make()
    al.add_item(d, "plate", "MH12AB1234")
    o = al.overview(d)
    assert o["unacknowledged"] == 1 and o["alerts"][0]["historical"] is True and "cam1" in o["alerts"][0]["title"]
    assert al.scan(d) == 0 and al.overview(d)["unacknowledged"] == 1                      # scanning again changes nothing


def test_new_sighting_raises_one_alert_and_only_once():
    d = make()
    al.add_item(d, "plate", "MH12AB1234")
    al.overview(d)
    grow(d, "cam3_vehicles.json", veh("cam3", 4, "MH12AB1234", 50, 55))
    assert al.scan(d) == 1
    a = [x for x in al.overview(d)["alerts"] if not x["historical"]]
    assert len(a) == 1 and a[0]["camera_id"] == "cam3" and a[0]["link"] == "#investigate/cam3/4"
    assert al.scan(d) == 0
    grow(d, "cam2_vehicles.json", veh("cam2", 2, "KA01XY9999"))                             # a different plate: no alert
    assert al.scan(d) == 0


def test_plate_matching_tolerates_one_ocr_error_but_not_two_and_accepts_a_prefix():
    d = make()
    al.add_item(d, "plate", "MH12AB1234")
    al.overview(d)
    grow(d, "cam2_vehicles.json", veh("cam2", 5, "MH12AB1284"))                             # one character off
    grow(d, "cam2_vehicles.json", veh("cam2", 6, "MH12AB1299"))                             # two off
    assert al.scan(d) == 1
    assert "One character differs" in [x for x in al.overview(d)["alerts"] if not x["historical"]][0]["detail"]
    al.add_item(d, "plate", "KA01")                                                         # the start of a plate
    grow(d, "cam3_vehicles.json", veh("cam3", 9, "KA01XY9999"))
    assert al.scan(d) == 1


def test_clothing_watch():
    d = make()
    it = al.add_item(d, "clothing", "red jacket")
    assert it["existing_hits"] == 1
    grow(d, "cam3_attributes.json", attrs("cam3", 5, "red", "black", 40, 50))
    grow(d, "cam3_attributes.json", attrs("cam3", 6, "green", "black", 41, 50))
    assert al.scan(d) == 1
    assert "cam3" in [x for x in al.overview(d)["alerts"] if not x["historical"]][0]["title"]


def test_unaccounted_gap_rule():
    d = make()
    assert al.add_item(d, "gap", "30")["existing_hits"] == 0                                # the cam2 -> cam3 gap is 22 s
    assert al.add_item(d, "gap", "15")["existing_hits"] == 1
    o = al.overview(d)
    a = [x for x in o["alerts"] if x["kind"] == "gap"][0]
    assert "22s unaccounted between cam2 and cam3" in a["title"] and a["link"] == "#investigate/cam2/1"


def test_camera_down_alerts_once_per_outage():
    d = make()
    live = d / "outputs" / "cam1_observations_live.json"
    old = time.time() - 120

    def age():
        os.utime(live, (old, old))
        al._SIG.clear()                                    # "down" depends on the clock, so ask again regardless of the cache

    live.write_text("[]")
    al.add_item(d, "camera_down")
    assert al.overview(d)["unacknowledged"] == 0           # it is up now
    age()
    assert al.scan(d) == 1                                 # the feed stopped
    age()
    assert al.scan(d) == 0                                 # still down: not told again
    live.write_text("[]")                                  # back up
    al._SIG.clear()
    assert al.scan(d) == 0
    age()
    assert al.scan(d) == 1                                 # a second outage is a new alert
    a = [x for x in al.overview(d)["alerts"] if x["kind"] == "camera_down"]
    assert len(a) == 2 and all("cam1 is down" == x["title"] and x["link"] == "#cameras" for x in a)


def test_acknowledge_one_and_all():
    d = make()
    al.add_item(d, "plate", "MH12AB1234"); al.add_item(d, "clothing", "green jacket")
    o = al.overview(d)
    assert o["unacknowledged"] == 3
    assert al.acknowledge(d, o["alerts"][0]["id"]) == 1 and al.overview(d)["unacknowledged"] == 2
    assert al.acknowledge(d, "nope") == 0
    assert al.acknowledge(d, everything=True) == 2 and al.overview(d)["unacknowledged"] == 0


def test_removing_an_item_stops_new_alerts_and_keeps_the_log():
    d = make()
    it = al.add_item(d, "plate", "MH12AB1234")
    al.overview(d)
    assert al.remove_item(d, it["id"]) is True and al.remove_item(d, it["id"]) is False
    grow(d, "cam3_vehicles.json", veh("cam3", 4, "MH12AB1234"))
    assert al.scan(d) == 0 and al.overview(d)["items"] == [] and len(al.overview(d)["alerts"]) == 1


def test_new_alerts_are_sent_to_the_webhook_but_old_ones_are_not():
    catcher = Catcher()
    try:
        d = make()
        os.environ["CATCHCAM_WEBHOOK_URL"] = catcher.url + "/hook"
        al.add_item(d, "plate", "MH12AB1234")
        al.overview(d); al.wait_for_notifications()
        assert catcher.got == []                                                            # already in the footage: not sent
        grow(d, "cam3_vehicles.json", veh("cam3", 4, "MH12AB1234", 50, 55))
        al.scan(d); al.wait_for_notifications()
        assert len(catcher.got) == 1
        path, body = catcher.got[0]
        assert path == "/hook" and "MH12AB1234" in body["text"] and body["content"] == body["text"] and body["alert"]["camera_id"] == "cam3"
        assert [a for a in al.overview(d)["alerts"] if not a["historical"]][0]["notification"] == "webhook: sent"
        assert al.overview(d)["notifications"] == {"webhook": True, "telegram": False}
    finally:
        os.environ.pop("CATCHCAM_WEBHOOK_URL", None); catcher.stop()


def test_telegram_goes_to_the_bot_api_and_the_token_is_never_echoed():
    catcher = Catcher()
    try:
        cfg = {"telegram_token": "123:SECRET", "telegram_chat_id": "42", "telegram_api_base": catcher.url}
        assert notify.deliver(cfg, "hello", {}) == "telegram: sent"
        assert catcher.got == [("/bot123:SECRET/sendMessage", {"chat_id": "42", "text": "hello"})]
        bad = notify.deliver({"telegram_token": "123:SECRET", "telegram_chat_id": "42", "telegram_api_base": "http://127.0.0.1:9"}, "x", {})
        assert bad.startswith("telegram: failed") and "SECRET" not in bad
    finally:
        catcher.stop()


def test_a_failing_webhook_is_recorded_not_raised_and_odd_schemes_are_refused():
    assert notify.deliver({"webhook_url": "http://127.0.0.1:9/x"}, "t", {}).startswith("webhook: failed")
    assert "only http(s)" in notify.deliver({"webhook_url": "file:///etc/passwd"}, "t", {})
    assert notify.deliver({}, "t", {}) == "not configured"
    d = make()
    (d / "alerts_config.json").write_text(json.dumps({"webhook_url": "http://example.invalid/h"}))
    assert notify.configured(notify.load_config(d)) == {"webhook": True, "telegram": False}


if __name__ == "__main__":
    import traceback
    ok = bad = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            al._SIG.clear()
            try:
                fn(); ok += 1; print("PASS", name)
            except Exception:
                bad += 1; print("FAIL", name); traceback.print_exc()
    print(f"{ok} passed, {bad} failed")
    raise SystemExit(1 if bad else 0)
