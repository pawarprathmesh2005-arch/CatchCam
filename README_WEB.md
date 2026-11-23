# CatchCam web interface

A faster replacement for the Streamlit dashboard: one web page served by your existing API. Three screens instead of ten.

| Screen | What it does |
|---|---|
| **Investigate** (home) | Your cameras as a video wall. One search box takes anything: "person in a green jacket", a plate, "track 5 on cam1", or a question ("where did he go next?"). Pick a person and the video **follows them from camera to camera**, with a timeline, a "not seen for 23s" banner at each gap, and tabs for Journey, Floor plan, Who was with them, Clothing and Gaps. The mic button takes spoken questions; the speaker button reads answers aloud. |
| **Cameras** | Every camera with its status. **Upload a video** (with a progress bar), press **Process**, and CatchCam finds the people, learns their appearance, matches them across cameras and reads clothing colours. **Live camera** takes an RTSP address or a webcam number. |
| **Insights** | Counts, when people appeared on each camera, journeys with unexplained gaps (one click to investigate), and matches where the clothing disagrees (possible wrong matches). |

## Install (2 minutes)

1. Unzip `catchcam_web.zip` into your project folder, next to `api.py`. Choose **Replace** if asked. It adds `web/`, `webapp/`, `routes_web.py` and `test_web.py`.
2. Open `api.py`, find this list and add `"routes_web"`:
   ```python
   OPTIONAL_ROUTE_MODULES = [
       "routes_health", "routes_vehicles", "routes_attributes",
       "routes_qa", "routes_floorplan", "routes_web",      # <- add this one
   ]
   ```
3. Start the API as usual: `uvicorn api:app --reload --port 8000`
4. Open **http://localhost:8000** in Chrome or Edge. (`http://localhost:8000/modules` should list `routes_web`.)

You no longer need `streamlit run dashboard.py`. The old dashboard still works if you want it.

## Try it without any footage

```
python -m webapp.demo_data demo_data --webm
$env:CATCHCAM_DATA_DIR = "demo_data"        # PowerShell. In cmd:  set CATCHCAM_DATA_DIR=demo_data
uvicorn api:app --port 8000
```
Three synthetic cameras, three people, one 23-second gap. Search "person in green jacket".

## Adding your own footage

Cameras → **Upload a video**, give it a name (cam1, cam2…), leave "Process it straight after uploading" ticked. Progress is shown stage by stage. Add every camera, then the people are matched across all of them (matching needs at least two cameras).

- Needs `ultralytics` (you have it). Appearance matching needs `torch` and `torchvision` (you have them: they are used by `extract_embeddings.py`). If something is missing the job says which step it skipped and why, instead of failing silently.
- If your cameras were **not recorded at the same moment**, open *Advanced* and enter the number of seconds after the first camera that this one starts.
- Browsers can only play H.264/VP8/VP9 video. If your file is another format, the page still shows it through a server-drawn stream (no scrubbing), and a **Convert for browser** button appears if `ffmpeg` is installed.

## Choosing which cameras to watch, and deleting a camera
- **Investigate → the bar above the wall**: `Show 1 / 2 / 4 / 6 / All` sets how many cameras are on screen at once; with more cameras than fit, `‹ ›` pages through them.
- Each camera has a **button**: click it to hide or show that camera on the wall (double-click to open just that one). `Select all` brings everything back. Your choice is remembered in the browser.
- Click a tile to look closer. While looking at one camera, the row of camera buttons above the picture switches to any other camera at once: no page refresh and no trip back to the wall. `← All cameras` returns.
- **Cameras → Delete** removes a camera's video and everything found in it (people, clothing, matches) after a confirmation. It is refused while that camera is being processed. The name can be used again afterwards.

## Making processing faster, or more sensitive (Cameras screen, "When processing")
- **Speed**: `Every frame` (slowest, most detailed), `Faster` (every 3rd frame, about 3x quicker) or `Fastest` (every 5th frame and a smaller picture). Fine for people walking; very quick or distant people may be missed. Try a short clip first.
- **Sensitivity**: `High` also counts people the detector is less sure about. Use it for dark, green night-vision or distant footage. It can add some false people.
- The choice is remembered and applies the next time you press Process.

## Camera start times (so journeys across cameras line up)
Every camera's own clock starts at 0. To compare cameras, tell CatchCam the clock shown on each camera's picture at the first frame of its video: on the **Cameras** screen type it in the **Starts at** box (for example `12:38:53`) and press Set. Everything already found is shifted, with no re-processing, and times then appear as real clock times (12:38:53). Do this for every camera in a journey.

## Fast processing on Google Colab (GPU)
1. On your PC, in the project folder: `python make_colab_bundle.py` makes `catchcam_project.zip` (code only).
2. Put it and a `clips` folder (cam1.mp4, cam2.mp4 ...) in a Google Drive folder called `CatchCamColab`.
3. Open `CatchCam_Colab.ipynb` in Colab (Runtime, T4 GPU), edit the start times in step 4, run the cells. Download `demo_data.zip`.
4. Unzip it into `D:\CatchCam`, then in PowerShell: `$env:CATCHCAM_DATA_DIR = "D:\CatchCam\demo_data"` and start CatchCam as usual. Your old data stays untouched in the normal folder.
`process_clips.py` can also run on your PC: `python process_clips.py clips --data demo_data --start cam1=12:38:53 --speed fast`.

## Why only 4 cameras play at once on the wall
A browser allows only about 6 open connections to one server, and every playing video keeps one open. With 10 cameras all playing, the last ones stayed black and search had to wait in the same queue. So the wall plays 4 cameras live and shows the others as pictures; press **▶ Live** on a picture to watch it (another camera goes back to a picture). Clicking any camera still opens it fully, and `Show 1 / 2 / 4` plays all that are shown.

## False detections and slow processing (Cameras screen)
- **Remove false detections** (button above the camera list): deletes "people" that were only blips: seen for under about a quarter of a second, or low-confidence and under a couple of seconds, or flickering on one spot. Long, confident tracks are always kept, including people sitting still. It also removes them from clothing data, appearance data and matches. The original is saved once as `outputs/<camera>_observations.before_cleanup.json`. It runs automatically at the end of every Process, so you only need to press it for cameras you processed earlier.
- **Empty footage: Skip it (faster)**: before detecting people, CatchCam finds the stretches where something moves and only analyses those. On a quiet camera that is about a third of the video. A camera that is busy all the time is analysed in full. Choose **Analyse everything** if you think people were missed. The job notes say how much was analysed.
- Processing leaves one CPU core free so the page and search stay responsive while it runs.
- Defaults are now: Speed = Faster (every 3rd frame), Sensitivity = Normal, Empty footage = Skip.

## Tests

```
python -m pytest -q test_e2e.py test_e2e_features.py test_web.py
```
`test_web.py` builds its own synthetic cameras in a temp folder and never touches your real data.

## What is not in the new interface yet

- **Incident reconstruction for a time window** (the old page's "all activity between t1 and t2") and **journey video export**. Both still work in the old dashboard and through the API (`/incident`, `/journey_video`). The planned PDF incident report and evidence-clip export will replace them.
- Live cameras are **shown**, but people are tracked live only if `realtime_track.py` is running for that camera.

## Honest limits

- Tested with synthetic cameras in a real browser (Chromium) against the API through a stand-in for FastAPI. Not yet run against your real FastAPI server or your real footage, so expect a small fix or two on first launch.
- Live view uses a server-drawn image stream: a few frames per second, fine for monitoring and demos, not a replacement for a surveillance player.
- Clothing colours are approximate and only reliable within one day.

---

# Accuracy check, incident report and watchlist alerts

They are loaded automatically by `routes_web`, so `api.py` needs no further change. Install the one new package: `pip install reportlab`.

## Accuracy check (Insights → Accuracy check)

How often is CatchCam right? You tell it who is who, and it scores its own cross-camera matching against your answers.

1. Open **Insights → Accuracy check**. Every tracked person appears with a picture.
2. Type the same name (`Asha`, `P1`...) under every track that is the same real person, on any camera. Leave tracks you can't identify blank: they are not scored.
3. Press **Run the check**.

You get precision (of the matches CatchCam made, how many were right), recall (of the real matches, how many it found), an F1 score, how many people's whole journeys were completely right, and a table that shows what would happen at different similarity thresholds, so you can choose the best one. The labels are saved in `eval_labels.json` in your data folder. **Start from CatchCam's guess** pre-fills the names from its own matches to save typing, but check them against the video: starting from the guess and not correcting it makes the score look better than it is.

Label at least 10 to 15 people across 2 or more cameras before quoting the numbers anywhere. Six tracks, as in the demo, only show that the check works.

## Incident report (Investigate → Export report (PDF))

Open any person on the Investigate screen and press **Export report (PDF)**. The PDF has a summary, a timeline with unaccounted gaps marked in red, the link-by-link table (with the appearance score and the clothing check), snapshots from each camera with the person boxed, who was seen alongside them, and the floor plan if you set one up. It is generated from your data and is evidence to review, not proof: the report says so itself. Without the footage it still works, just without snapshots.

## Watchlist alerts (Insights → Watchlist and alerts, and the bell in the top bar)

Add something to watch for: a **number plate** (or its start, e.g. `MH12`), a **clothing description** (`red jacket`), a **journey gap** longer than N seconds, or **a camera going down**. When new footage is processed and something matches, an alert appears in the list and on the bell. Things already in the footage when you add an item are shown too, marked *already in the footage*. **Investigate** opens the person; **Acknowledge** clears an alert.

To also get alerts on your phone, set one or both of these before starting the API (they can't be changed through the web page on purpose, so nobody who can open the dashboard can point alerts at an address of their choosing):

```
# PowerShell
$env:CATCHCAM_WEBHOOK_URL    = "https://hooks.slack.com/services/..."   # Slack, Discord, Teams, n8n, Zapier...
$env:CATCHCAM_TELEGRAM_TOKEN = "123456:ABC..."                         # from @BotFather
$env:CATCHCAM_TELEGRAM_CHAT  = "your chat id"                           # message your bot once, then open https://api.telegram.org/bot<token>/getUpdates
```
You can use `alerts_config.json` in the data folder instead. Sending to Slack or Telegram has not been tried against the real services; the check is done with stand-ins, so send yourself a test alert first.

## Tests
```
python -m pytest -q test_e2e.py test_e2e_features.py test_web.py test_tools.py
python -m evaluation.test_evaluation
python -m alerts.test_alerts
```

## Following one person across cameras (v10)
1. Tell CatchCam when each camera starts (without it all cameras share a clock starting at 0 and the order is meaningless):
   `python -m webapp.set_times --data demo_data cam1=12:38:22 cam2=12:46:28 cam3=hh:mm:ss ...`
2. Open the person in Investigate (search `track 15 on cam4`), then the **Find on other cameras** tab: best candidates per camera, strongest first. Click Watch to confirm each.
Ranking = appearance (average of all people subtracted first) + clothing colours (gray/silver/white count as one washed-out family) + time feasibility. API: `GET /web/follow?camera_id=&track_id=`.
