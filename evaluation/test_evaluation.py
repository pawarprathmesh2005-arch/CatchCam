"""Tests for the accuracy evaluation. Builds tiny synthetic data in a temp folder (no video needed).
Run:  python -m pytest -q evaluation/test_evaluation.py     or     python evaluation/test_evaluation.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import core as ev  # noqa: E402

# Asha: cam1#1 -> cam2#1 -> cam3#1 | Ravi: cam1#2 -> cam2#2 | Meena: cam3#2 (one camera only)
TRACKS = {"cam1": [(1, 2, 14), (2, 4, 16)], "cam2": [(1, 18, 30), (2, 26, 38)], "cam3": [(1, 52, 59), (2, 10, 28)]}
GOOD_MATCHES = [("cam1", 1, "cam2", 1, 0.91), ("cam2", 1, "cam3", 1, 0.84), ("cam1", 2, "cam2", 2, 0.88)]
TRUTH = {("cam1", 1): "Asha", ("cam2", 1): "Asha", ("cam3", 1): "Asha", ("cam1", 2): "Ravi", ("cam2", 2): "Ravi", ("cam3", 2): "Meena"}


def make(matches=GOOD_MATCHES, embeddings=None, labels=TRUTH):
    d = Path(tempfile.mkdtemp(prefix="catchcam_eval_"))
    (d / "outputs").mkdir()
    (d / "matches").mkdir()
    for cam, rows in TRACKS.items():
        obs = []
        for tid, a, b in rows:
            for k in range(5):
                t = a + (b - a) * k / 4
                obs.append({"camera_id": cam, "track_id": tid, "frame": int(t * 8), "timestamp": t, "class": "person", "confidence": 0.9,
                            "bbox_xyxy": [10, 10, 60, 130]})
        (d / "outputs" / f"{cam}_observations.json").write_text(json.dumps(obs))
    (d / "matches" / "cross_camera_matches.json").write_text(json.dumps(
        [{"camera_a": a, "track_a": ta, "camera_b": b, "track_b": tb, "similarity": s} for a, ta, b, tb, s in matches]))
    if embeddings:
        (d / "embeddings").mkdir()
        for cam in TRACKS:
            tr = {str(t): {"embedding": v} for (c, t), v in embeddings.items() if c == cam}
            (d / "embeddings" / f"{cam}_embeddings.json").write_text(json.dumps({"camera_id": cam, "tracks": tr}))
    if labels:
        ev.save_labels(d, [{"camera_id": c, "track_id": t, "person": p} for (c, t), p in labels.items()])
    return d


# well separated "appearance" vectors: each person has a direction, tracks of one person are near it
PERSON_VEC = {"Asha": [1, 0, 0], "Ravi": [0, 1, 0], "Meena": [0, 0, 1]}


def emb_for(truth, wobble=0.05, swap=None):
    out = {}
    for i, (k, p) in enumerate(sorted(truth.items())):
        who = (swap or {}).get(k, p)
        v = list(PERSON_VEC[who])
        v[(i + 1) % 3] += wobble * (i % 3 + 1)
        out[k] = v
    return out


def test_perfect_matching_scores_perfectly():
    r = ev.evaluate(make())
    assert r["status"] == "ok" and r["labelled_tracks"] == 6 and r["people"] == 3
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["pairs"]["true_same"] == 4 and r["pairs"]["false_matches"] == 0 and r["pairs"]["missed_matches"] == 0
    assert r["journeys"] == {**r["journeys"], "with_journey": 2, "fully_correct": 2}     # Meena has one camera: no journey to rebuild
    assert r["fragmentation"] == []


def test_wrong_and_missing_links_are_counted_and_listed():
    # lose Asha's cam2->cam3 hop, and wrongly link Ravi (cam2 #2) to Meena (cam3 #2)
    d = make(matches=[("cam1", 1, "cam2", 1, 0.91), ("cam1", 2, "cam2", 2, 0.88), ("cam2", 2, "cam3", 2, 0.77)])
    r = ev.evaluate(d)
    p = r["pairs"]
    # truth pairs across cameras: A1-A2, A1-A3, A2-A3, R1-R2 = 4.  CatchCam linked A1-A2, R1-R2 and R2-M
    # (Meena is on cam3 from 10s, before Ravi leaves cam2, so the graph points the edge cam3 -> cam2: R1-M is not linked)
    assert p["correct_matches"] == 2 and p["missed_matches"] == 2 and p["false_matches"] == 1
    assert abs(r["precision"] - 2 / 3) < 1e-9 and r["recall"] == 0.5
    assert {m["a_person"] + "/" + m["b_person"] for m in r["false_match_list"]} == {"Ravi/Meena"}
    assert {m["person"] for m in r["missed_match_list"]} == {"Asha"}
    assert r["journeys"]["fully_correct"] == 0
    bad = {j["person"]: j for j in r["journeys"]["details"]}
    assert bad["Ravi"]["extra"] == ["cam3 #2 (Meena)"] and bad["Asha"]["missing"] == ["cam3 #1"]


def test_only_labelled_tracks_are_scored():
    partial = {k: v for k, v in TRUTH.items() if k[1] == 1}                     # label Asha's three tracks only
    r = ev.evaluate(make(labels=partial))
    assert r["labelled_tracks"] == 3 and r["pairs"]["checked"] == 3 and r["precision"] == 1.0
    assert any("Only 3 tracks" in n for n in r["notes"])


def test_need_labels_message():
    d = make(labels=None)
    r = ev.evaluate(d)
    assert r["status"] == "need_labels" and "Label at least two" in r["message"]
    assert "Label at least two" in ev.to_markdown(r)


def test_labels_are_validated():
    d = make(labels=None)
    for rows, text in [([{"camera_id": "cam9", "track_id": 1, "person": "A"}], "does not exist"),
                       ([{"camera_id": "cam1", "track_id": 1, "person": "<b>x</b>"}], "not a usable person name"),
                       ([{"camera_id": "cam1", "person": "A"}], "needs camera_id")]:
        try:
            ev.save_labels(d, rows)
        except ValueError as e:
            assert text in str(e), e
        else:
            raise AssertionError("should have been refused: " + text)
    assert ev.save_labels(d, [{"camera_id": "cam1", "track_id": 1, "person": "A"}, {"camera_id": "cam1", "track_id": 2, "person": ""}]) == 1
    assert ev.load_labels(d) == {("cam1", 1): "A"}                               # an empty person means "unlabelled"
    assert not list(d.glob("*.tmp"))


def test_ranking_and_threshold_sweep_with_embeddings():
    r = ev.evaluate(make(embeddings=emb_for(TRUTH)))
    k = r["ranking"]
    assert k["rank1"] == 1.0 and k["map"] == 1.0 and k["queries"] == 5            # Meena has nobody to find on another camera
    sweep = {s["threshold"]: s for s in r["threshold_sweep"]}
    assert sweep[0.5]["recall"] == 1.0 and sweep[0.5]["precision"] == 1.0
    strict = ev.threshold_sweep(ev.load_labels(make(embeddings=emb_for(TRUTH))), ev.load_embeddings(make(embeddings=emb_for(TRUTH))), steps=(0.99999,))[0]
    assert strict["recall"] < 1.0 and strict["missed"] > 0                         # a cut-off above the real similarities misses people


def test_ranking_notices_a_confused_appearance():
    swapped = emb_for(TRUTH, swap={("cam2", 1): "Ravi"})                           # Asha on cam2 looks like Ravi
    k = ev.evaluate(make(embeddings=swapped))["ranking"]
    assert k["rank1"] < 1.0 and k["map"] < 1.0


def test_fragmentation_is_reported_not_counted_as_a_match_error():
    labels = {**TRUTH, ("cam1", 2): "Asha"}                                         # tracker split Asha into two tracks on cam1
    r = ev.evaluate(make(labels=labels))
    assert r["fragmentation"] == [{"person": "Asha", "camera_id": "cam1", "tracks": [1, 2]}]


def test_track_list_groups_predicted_journeys_and_carries_labels():
    d = make()
    t = ev.track_list(d)
    assert t["labelled"] == 6 and t["people"] == ["Asha", "Meena", "Ravi"]
    g = {(r["camera_id"], r["track_id"]): r["predicted_group"] for r in t["tracks"]}
    assert g[("cam1", 1)] == g[("cam2", 1)] == g[("cam3", 1)] and g[("cam1", 2)] == g[("cam2", 2)] != g[("cam1", 1)]
    assert t["tracks"][0]["thumb_url"].startswith("/web/thumb?camera_id=")


def test_markdown_report_has_the_numbers():
    md = ev.to_markdown(ev.evaluate(make(embeddings=emb_for(TRUTH))))
    for s in ("Precision | 100.0%", "Rank-1 | 100.0%", "Similarity threshold sweep", "not measured here"):
        assert s in md, s
    assert (Path(make()) / ev.GT_FILE).exists()


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
