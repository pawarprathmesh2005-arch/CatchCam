# CatchCam: clothing attributes, conversational Q&A, floor-plan fusion

Unzip into your project folder (next to api.py). `api.py` already auto-loads `routes_attributes`, `routes_qa` and
`routes_floorplan`. Restart the API and open http://localhost:8000/modules: all three should be listed.
`dashboard.py` is replaced (it adds the pages Ask CatchCam, Clothing Search, Floor Plan on top of everything you had).
`vehicle_intel/core.py` is replaced too (one small backward-compatible change that Clothing Search uses).

## 1. Clothing attributes (no face recognition)
    python -m person_attrs.extract --video cameras/cam1_topleft.mp4 --observations outputs/cam1_observations.json --camera_id cam1 --out_dir outputs
Run once per camera -> `outputs/cam1_attributes.json` (top colour and bottoms colour per person, voted over up to 8 frames).
- GET /attributes/search?description=person in a red jacket and blue jeans
- GET /attributes?upper_color=red
- GET /attributes/verify_matches   marks cross-camera Re-ID matches whose clothes completely differ ("conflict") so you can review them first
Limits: coarse colour names only (11 colours); CCTV lighting shifts colours; clothing is same-day evidence, not an identity.

## 2. Conversational Q&A (typed or spoken)
GET /qa?question=...   (dashboard page: Ask CatchCam). Examples: "Where did the person in the red jacket go?", "Where did track 5 on
cam1 come from?", "Who was with track 3 on cam2?", "How many people were seen between 10 and 60 seconds?", "Are there any blind spots in
the journey of track 1 on cam1?", "What was track 4 on cam2 wearing?", "Which cameras are down?", "Where did the car MH12AB1234 go?"
Follow-ups work: after an answer about a person, "What was he wearing?" / "And where did he come from?" refer to the same person.
- Every answer is built from your own data (journeys, gaps, relationships, clothing, vehicles, camera health), never invented.
  If it cannot tell who you mean, or nobody matches, it says so.
- Voice in: the mic button (Chrome/Edge). Voice out: the "Read answer aloud" button (browser speech, no API key). It is a button because
  browsers block speech that starts without a click.
Limits: it understands the question types above (rule-based, not free chat); unusual wording gets "I didn't understand" plus examples.
A language model could be added later purely to rephrase answers.

## 3. 2D floor-plan fusion
1. Create `floorplan.json` in your project folder (see `floorplan/floorplan.example.json`): plan size, each camera's position.
2. To get accurate paths (not just a pin per camera), calibrate each camera with 4-8 floor points:
       python -m floorplan.grid_still --video cameras/cam1_topleft.mp4 --out cam1_grid.png
   Read the pixel coordinates of floor landmarks from the grid picture, measure the same spots on your plan, put both lists in floorplan.json.
       python -m floorplan.core check floorplan.json      # reports whether your clicks are consistent
3. Dashboard: Floor Plan page, or GET /floorplan/render?camera_id=cam1&track_id=5 (PNG), /floorplan/journey (JSON), /floorplan/config.
Unexplained gaps (missing-camera detection) are drawn as dashed amber/red lines. Uncalibrated cameras appear as pins marked "approx.".
Limits: assumes a flat floor; positions are the bottom-centre of the person's box, so they are wrong when the feet are cut off by the frame
edge or hidden behind objects; accuracy depends entirely on calibration quality.

## Tests (no GPU, no model downloads)
    pytest -q test_e2e.py test_e2e_features.py
    python -m unittest person_attrs.test_person_attrs floorplan.test_floorplan
