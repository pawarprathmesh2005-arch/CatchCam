"""
Background jobs for the web interface: processing uploaded footage and converting videos for the browser.

Processing = the same steps you would run by hand, one after another:
  1. track people (YOLO + ByteTrack)            -> outputs/<cam>_observations.json
  2. appearance embeddings (your extract_embeddings.py)   -> embeddings/<cam>_embeddings.json
  3. match people across all cameras (your match_cameras.py) -> matches/cross_camera_matches.json
  4. clothing colours (person_attrs)             -> outputs/<cam>_attributes.json
  5. browser-friendly copy of the video (ffmpeg, if installed)

A missing optional package never breaks the job: that step is reported as skipped with the reason.
"""
import threading
import time
import traceback
import uuid
from pathlib import Path

from webapp import core

STAGES = [("track", "Finding people in the video", 0.50), ("embed", "Learning what each person looks like", 0.15),
          ("match", "Matching people across cameras", 0.10), ("clothes", "Reading clothing colours", 0.10),
          ("prepare", "Preparing the video for your browser", 0.15)]


class Job:
    def __init__(self, kind, camera_id):
        self.id = uuid.uuid4().hex[:10]
        self.kind, self.camera_id = kind, camera_id
        self.state, self.stage, self.percent = "queued", "Waiting to start", 0.0
        self.notes, self.error = [], None
        self.started, self.finished = time.time(), None

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "camera_id": self.camera_id, "state": self.state, "stage": self.stage,
                "percent": round(self.percent * 100), "notes": self.notes, "error": self.error,
                "seconds": round((self.finished or time.time()) - self.started, 1)}


class JobManager:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def start(self, kind, camera_id, fn):
        with self.lock:
            if any(j.state in ("queued", "running") and j.camera_id == camera_id for j in self.jobs.values()):
                raise RuntimeError(f"{camera_id} is already being processed.")
            job = Job(kind, camera_id)
            self.jobs[job.id] = job

        def run():
            job.state = "running"
            try:
                fn(job)
                job.state, job.percent, job.stage = "done", 1.0, "Finished"
            except Exception as e:                                           # shown to the user, never swallowed
                job.state, job.error = "error", str(e) or e.__class__.__name__
                job.notes.append(traceback.format_exc().strip().splitlines()[-1])
            finally:
                job.finished = time.time()

        threading.Thread(target=run, daemon=True).start()
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def recent(self, n=10):
        return sorted(self.jobs.values(), key=lambda j: -j.started)[:n]


def _begin(job, index, label=None):
    job.stage = label or STAGES[index][1]
    job.percent = sum(s[2] for s in STAGES[:index])


def _progress(job, index):
    base, weight = sum(s[2] for s in STAGES[:index]), STAGES[index][2]
    return lambda frac: setattr(job, "percent", base + weight * max(0.0, min(frac, 1.0)))


SPEEDS = {"normal": dict(every=1, imgsz=None), "fast": dict(every=3, imgsz=None), "fastest": dict(every=5, imgsz=480)}
SENSITIVITY = {"normal": 0.35, "high": 0.2}


def process_footage(data_dir, camera_id, job, weights="yolov8n.pt", time_offset=0.0, speed="normal", sensitivity="normal", skip_empty=True):
    data_dir = Path(data_dir)
    video = core.camera_videos(data_dir).get(camera_id)
    if video is None:
        raise FileNotFoundError(f"No video is registered for {camera_id}.")
    out_dir = data_dir / "outputs"
    if time_offset:
        offs = core._read_json(data_dir / "time_offsets.json", {})
        offs[camera_id] = time_offset
        core._write_json(data_dir / "time_offsets.json", offs)

    # 1. tracking (required)
    _begin(job, 0)
    from webapp import tracker
    obs_file = tracker.run(str(video), camera_id, str(out_dir), weights=weights, time_offset=time_offset, progress=_progress(job, 0),
                           conf=SENSITIVITY.get(sensitivity, 0.35), skip_empty=bool(skip_empty), notes=job.notes,
                           **SPEEDS.get(speed, SPEEDS["normal"]))
    from webapp import cleaning
    cleaned = cleaning.clean_camera(data_dir, camera_id)
    n_tracks = cleaned["tracks_after"]
    removed = cleaned["tracks_before"] - cleaned["tracks_after"]
    job.notes.append(f"Found {n_tracks} people in {camera_id}." + (f" Ignored {removed} false detections (blips that lasted under a second, "
                     f"or were low-confidence and did not move)." if removed else ""))

    # 2 + 3. embeddings and cross-camera matching (needs torch + CLIP; skipped with a reason if absent)
    _begin(job, 1)
    try:
        from extract_embeddings import extract_embeddings
        extract_embeddings(str(video), obs_file, camera_id, out_dir=str(data_dir / "embeddings"))
        _begin(job, 2)
        from match_cameras import match_cameras
        emb_files = sorted((data_dir / "embeddings").glob("*_embeddings.json"))
        if len(emb_files) >= 2:
            match_cameras([str(f) for f in emb_files], out_dir=str(data_dir / "matches"))
        else:
            job.notes.append("Only one camera has footage so far, so there is nothing to match yet. Add a second camera.")
            _ensure_matches_file(data_dir)
    except ImportError as e:
        job.notes.append(f"Skipped appearance matching: {e.name or e} is not installed. People are tracked, but not linked across cameras.")
        _ensure_matches_file(data_dir)
    except Exception as e:                                       # noqa: BLE001  e.g. the model could not be downloaded (no internet)
        # the tracking above already worked and is kept; say plainly what failed instead of failing the whole job
        job.notes.append(f"Appearance matching failed: {str(e) or e.__class__.__name__}. People are tracked, but not linked across cameras. "
                         f"Fix the problem and press Process again.")
        _ensure_matches_file(data_dir)

    # 4. clothing (optional)
    _begin(job, 3)
    try:
        from person_attrs.extract import extract as extract_clothing
        extract_clothing(str(video), obs_file, camera_id, out_dir=str(out_dir))
    except ImportError as e:
        job.notes.append(f"Skipped clothing colours: {e.name or e} is not installed.")

    # 5. playback copy
    _begin(job, 4)
    _prepare_playback(data_dir, camera_id, job)


def _ensure_matches_file(data_dir):
    f = Path(data_dir) / "matches" / "cross_camera_matches.json"
    if not f.exists():
        core._write_json(f, [])                      # the journey code expects the file to exist, even with no matches


def _prepare_playback(data_dir, camera_id, job):
    path, ok = core.playable_video(data_dir, camera_id)
    if ok:
        return
    try:
        if core.make_web_copy(data_dir, camera_id):
            job.notes.append("Converted the video so it plays in the browser.")
        else:
            job.notes.append("This video's format can't play in a browser and ffmpeg isn't installed, so the dashboard will "
                             "show it through the server stream instead (works, but no scrubbing).")
    except RuntimeError as e:
        job.notes.append(str(e))


def convert_only(data_dir, camera_id, job):
    job.stage = "Converting the video for your browser"
    if not core.make_web_copy(data_dir, camera_id):
        raise RuntimeError("ffmpeg is not installed. Install it (https://ffmpeg.org) or keep using the server stream.")
    job.percent = 1.0


MANAGER = JobManager()
