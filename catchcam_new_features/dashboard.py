"""
CatchCam — Dashboard
========================
The visual frontend for your project. Talks to the FastAPI backend
(api.py) over HTTP -- so make sure that's running first.

Usage:
    # Terminal 1:
    uvicorn api:app --reload --port 8000

    # Terminal 2:
    streamlit run dashboard.py

Then open the URL Streamlit prints (usually http://localhost:8501).
The .streamlit/config.toml file next to this script sets the color theme --
keep that folder alongside dashboard.py.

Requires: streamlit, requests (pip install streamlit requests)
"""
import html
import json
import os
import time

import requests
import streamlit as st
import streamlit.components.v1 as components

API_BASE = os.environ.get("CATCHCAM_API_URL", "http://localhost:8000")

st.set_page_config(page_title="CatchCam", layout="wide", page_icon="🎥")

# --- Visual identity: fonts, hairline borders, sharp corners, monospace data.
#     Design intent: this should feel like an investigator's evidence-review
#     tool, not a generic analytics SaaS dashboard. --------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

/* Monospace for anything data-like: timestamps, IDs, code blocks */
code, pre, .stCode, [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Sharp corners, hairline borders instead of soft SaaS shadows */
div[data-testid="stMetric"], div[data-testid="stExpander"],
div[data-testid="stDataFrame"], .stAlert {
    border: 1px solid #2A2F37 !important;
    border-radius: 2px !important;
    box-shadow: none !important;
}

div[data-testid="stMetric"] {
    background: #1A1D22;
    padding: 14px 16px;
}

/* Sidebar: slightly darker, clear separation */
section[data-testid="stSidebar"] {
    background: #101215;
    border-right: 1px solid #2A2F37;
}

/* Buttons: sharp corners, no shadow, amber accent already from theme */
.stButton button {
    border-radius: 2px !important;
    font-weight: 500;
}

/* Header rule under the title -- evokes a console/terminal divider, not a
   generic app title */
.catchcam-header {
    border-bottom: 1px solid #2A2F37;
    padding-bottom: 12px;
    margin-bottom: 20px;
}
.catchcam-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #4FB8A8;
    letter-spacing: 0.05em;
}

/* Status pill for the live connection indicator */
.status-live {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #4FB8A8;
    border: 1px solid #4FB8A8;
    padding: 2px 8px;
    border-radius: 2px;
}
.status-down {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #E05252;
    border: 1px solid #E05252;
    padding: 2px 8px;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)


def api_get(path, params=None, timeout=30):
    """Small wrapper so every API call gets the same error handling."""
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            detail = resp.json().get("detail", resp.text)
            return None, f"API error ({resp.status_code}): {detail}"
    except requests.exceptions.ConnectionError:
        return None, (
            f"Could not reach the API at {API_BASE}. "
            f"Is it running? (uvicorn api:app --reload --port 8000)"
        )
    except Exception as e:
        return None, f"Unexpected error: {e}"


def api_post(path, params=None, timeout=120):
    """POST version of api_get, for endpoints that trigger real work (like video generation)."""
    try:
        resp = requests.post(f"{API_BASE}{path}", params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            detail = resp.json().get("detail", resp.text)
            return None, f"API error ({resp.status_code}): {detail}"
    except requests.exceptions.ConnectionError:
        return None, f"Could not reach the API at {API_BASE}. Is it running?"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def api_get_bytes(path, params=None, timeout=60):
    """GET that returns raw bytes (used for the floor-plan picture)."""
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.content, None
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return None, f"API error ({resp.status_code}): {detail}"
    except requests.exceptions.ConnectionError:
        return None, f"Could not reach the API at {API_BASE}. Is it running?"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def speak_button(text, label="🔊 Read answer aloud"):
    """
    Reads `text` aloud with the browser's built-in speech synthesis (free, no backend). It is a button on purpose:
    browsers block speech that starts without a click.
    """
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(f"""
      <button id="b" style="background:#1A1D22;color:#E8A33D;border:1px solid #E8A33D;border-radius:2px;
              padding:8px 16px;font-size:14px;cursor:pointer;font-weight:500;">{label}</button>
      <script>
        const text = {payload};
        const b = document.getElementById('b');
        b.onclick = () => {{
          if (!('speechSynthesis' in window)) {{ b.innerText = 'Speech needs Chrome, Edge or Safari'; return; }}
          window.speechSynthesis.cancel();
          const u = new SpeechSynthesisUtterance(text);
          u.lang = 'en-US';
          window.speechSynthesis.speak(u);
        }};
      </script>""", height=50)


def similarity_badge(score):
    """Color-coded confidence indicator: teal=confident, amber=uncertain, red=weak."""
    if score >= 0.8:
        color = "#4FB8A8"
    elif score >= 0.6:
        color = "#E8A33D"
    else:
        color = "#E05252"
    return f'<span style="font-family:\'IBM Plex Mono\',monospace;color:{color};border:1px solid {color};padding:1px 6px;border-radius:2px;font-size:12px;">{score:.2f}</span>'


def voice_input_button():
    """
    A push-to-talk mic button using the browser's built-in speech
    recognition (Web Speech API) -- free, no backend, works in
    Chrome/Edge. Uses Streamlit's v2 component API, which runs
    unsandboxed (unlike components.html()'s iframe), so it can report
    the transcript straight back to Python without any page-navigation
    workaround.
    """
    voice_component = st.components.v2.component(
        "catchcam_voice_input",
        html="""
            <div>
              <button id="micBtn">🎤 Push to talk</button>
              <span id="micStatus"></span>
            </div>
        """,
        css="""
            button {
              background:#1A1D22; color:#E8A33D; border:1px solid #E8A33D;
              border-radius:2px; padding:8px 16px; font-size:14px; cursor:pointer;
              font-weight:500;
            }
            button:hover { background:#22262D; }
            #micStatus { margin-left:10px; color:#8B929E; font-size:13px; }
        """,
        js="""
            export default function(component) {
                const { setTriggerValue, parentElement } = component;
                const btn = parentElement.querySelector('#micBtn');
                const status = parentElement.querySelector('#micStatus');
                btn.onclick = () => {
                    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    if (!SpeechRecognition) {
                        status.innerText = "Speech recognition needs Chrome or Edge.";
                        return;
                    }
                    const recognition = new SpeechRecognition();
                    recognition.lang = 'en-US';
                    recognition.interimResults = false;
                    recognition.maxAlternatives = 1;
                    recognition.onstart = () => { status.innerText = "Listening..."; };
                    recognition.onerror = (event) => { status.innerText = "Error: " + event.error; };
                    recognition.onresult = (event) => {
                        const transcript = event.results[0][0].transcript;
                        status.innerText = 'Heard: "' + transcript + '"';
                        setTriggerValue('transcript', transcript);
                    };
                    recognition.start();
                };
            }
        """,
    )
    result = voice_component(on_transcript_change=lambda: None)
    return result.transcript if result and result.transcript else None


# --- Sidebar: connection status, always visible -----------------------------
with st.sidebar:
    st.markdown(
        '<div class="catchcam-eyebrow">MULTI-CAMERA REID SYSTEM</div>'
        '<h2 style="margin-top:4px;">🎥 CatchCam</h2>',
        unsafe_allow_html=True,
    )

    health, err = api_get("/health")
    if health:
        st.markdown(
            f'<span class="status-live">● LIVE</span> &nbsp; '
            f'{health["observation_files_found"]} camera files indexed',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="status-down">● NO CONNECTION</span>', unsafe_allow_html=True)
        st.caption(err)
        st.caption("Start the backend: `uvicorn api:app --reload --port 8000`")

    st.divider()
    page = st.radio(
        "Navigate",
        ["Cameras", "Journey Lookup", "Relationships", "Incident Reconstruction", "Natural Language Search", "Ask CatchCam", "Clothing Search", "Vehicles", "Floor Plan", "Camera Health"],
        label_visibility="collapsed",
    )


def page_header(eyebrow, title, caption):
    st.markdown(
        f'<div class="catchcam-header">'
        f'<div class="catchcam-eyebrow">{eyebrow}</div>'
        f'<h1 style="margin:4px 0 4px 0;">{title}</h1>'
        f'<div style="color:#8B929E;">{caption}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# --- Page: Cameras overview --------------------------------------------------
if page == "Cameras":
    page_header("FEED STATUS", "Cameras", "Every camera the system currently has tracking data for")
    data, err = api_get("/cameras")
    if err:
        st.error(err)
    elif not data["cameras"]:
        st.warning("No camera data found. Run track.py on at least one video first.")
    else:
        cols = st.columns(len(data["cameras"]))
        for col, cam in zip(cols, data["cameras"]):
            with col:
                st.metric(cam["camera_id"].upper(), f"{cam['num_tracks']} tracks")
                with st.expander("Track IDs"):
                    st.code(", ".join(str(t) for t in cam["track_ids"]), language=None)


# --- Page: Journey Lookup ----------------------------------------------------
elif page == "Journey Lookup":
    page_header("TRACE", "Journey Lookup", "Where a specific tracked person went — or came from")

    cameras_data, cam_err = api_get("/cameras")
    camera_options = [c["camera_id"] for c in cameras_data["cameras"]] if cameras_data else []

    col1, col2, col3 = st.columns(3)
    with col1:
        camera_id = st.selectbox("Camera", camera_options) if camera_options else st.text_input("Camera ID")
    with col2:
        track_id = st.number_input("Track ID", min_value=0, step=1)
    with col3:
        direction = st.radio("Direction", ["forward", "backward"], horizontal=True,
                              help="forward = where did they go? backward = where did they come from?")

    with st.expander("Coverage-gap settings"):
        max_gap = st.number_input("Normal time between cameras (s)", min_value=1.0, value=15.0, step=1.0,
                                   help="Longer unexplained gaps between two cameras are flagged as possible blind spots")

    col_trace, col_gaps, col_video = st.columns(3)
    with col_trace:
        trace_clicked = st.button("Trace Journey", type="primary")
    with col_gaps:
        gaps_clicked = st.button("🔍 Check Coverage Gaps")
    with col_video:
        video_clicked = st.button("🎬 Build Journey Video")

    if trace_clicked:
        data, err = api_get("/journey", params={
            "camera_id": camera_id, "track_id": track_id, "direction": direction,
        })
        if err:
            st.error(err)
        else:
            journey = data["journey"]
            if len(journey) <= 1:
                st.warning("No further hops found — this may be an isolated sighting "
                           "with no confident cross-camera match.")
            for i, hop in enumerate(journey):
                marker = "📍" if i == 0 else "↳"
                st.markdown(
                    f"{marker} **{hop['camera_id'].upper()}** "
                    f"<span style='font-family:\"IBM Plex Mono\",monospace;color:#8B929E;'>"
                    f"track {hop['track_id']}</span> — "
                    f"<span style='font-family:\"IBM Plex Mono\",monospace;'>"
                    f"{hop['first_seen_timestamp']:.2f}s → {hop['last_seen_timestamp']:.2f}s</span>",
                    unsafe_allow_html=True,
                )

    if gaps_clicked:
        data, err = api_get("/coverage_gaps", params={
            "camera_id": camera_id, "track_id": track_id, "max_normal_gap": max_gap,
        })
        if err:
            st.error(err)
            if "404" in str(err) and "Not Found" in str(err):
                st.info("The health module did not load. Check http://localhost:8000/modules and the API terminal.")
        else:
            st.caption("Full journey (traced back to first appearance, then forward)")
            for i, hop in enumerate(data["journey"]):
                st.markdown(
                    f"{'📍' if i == 0 else '↳'} **{hop['camera_id'].upper()}** "
                    f"<span style='font-family:\"IBM Plex Mono\",monospace;'>"
                    f"{hop['first_seen_timestamp']:.2f}s → {hop['last_seen_timestamp']:.2f}s</span>",
                    unsafe_allow_html=True,
                )
            st.write("")
            if not data["gaps"]:
                st.markdown(
                    f'<span class="status-live">● NO BLIND SPOTS</span> &nbsp; every hop is within {max_gap:.0f}s',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="status-down">● {len(data["gaps"])} UNEXPLAINED GAP(S)</span>',
                    unsafe_allow_html=True,
                )
                for g in data["gaps"]:
                    color = "#E05252" if g["severity"] == "critical" else "#E8A33D"
                    st.markdown(
                        f'<div style="border-left:3px solid {color};padding:6px 12px;margin:8px 0;">'
                        f'<span style="font-family:\'IBM Plex Mono\',monospace;color:{color};">'
                        f'{g["gap_seconds"]:.1f}s &nbsp; {g["severity"].upper()}</span><br>'
                        f'<b>{g["from_camera"].upper()}</b> (last seen {g["last_seen"]:.1f}s) → '
                        f'<b>{g["to_camera"].upper()}</b> (next seen {g["next_seen"]:.1f}s)<br>'
                        f'<span style="color:#8B929E;">{g["flag"]}</span></div>',
                        unsafe_allow_html=True,
                    )

    if video_clicked:
        with st.spinner("Trimming and stitching camera footage... this can take a minute"):
            data, err = api_post("/journey_video", params={"camera_id": camera_id, "track_id": track_id})
        if err:
            st.error(err)
            if "camera_videos.json" in str(err):
                st.info(
                    "Create a camera_videos.json file in your project folder mapping each "
                    "camera_id to its video file, e.g.:\n\n"
                    '```json\n{"cam1": "cameras/cam1_topleft.mp4", "cam2": "cameras/cam2_topright.mp4"}\n```'
                )
        else:
            st.success(f"Reconstructed a {len(data['journey'])}-camera journey")
            video_bytes_resp = requests.get(f"{API_BASE}{data['video_url']}", timeout=60)
            st.video(video_bytes_resp.content)


# --- Page: Relationships ------------------------------------------------------
elif page == "Relationships":
    page_header("PROXIMITY", "Relationships", "People detected together for a sustained period")

    min_duration = st.slider("Minimum duration together (seconds)", 0.0, 10.0, 1.0, 0.5)
    data, err = api_get("/relationships", params={"min_duration": min_duration})

    if err:
        st.error(err)
    elif not data["relationships"]:
        st.info("No relationships found at this threshold. Try lowering it, "
                "or run relationship_tracking.py first.")
    else:
        st.dataframe(
            data["relationships"],
            column_config={
                "camera_id": "Camera",
                "track_a": "Person A (track)",
                "track_b": "Person B (track)",
                "start_timestamp": st.column_config.NumberColumn("Start (s)", format="%.2f"),
                "end_timestamp": st.column_config.NumberColumn("End (s)", format="%.2f"),
                "duration_seconds": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
                "avg_proximity": st.column_config.NumberColumn("Avg Proximity", format="%.2f"),
            },
            use_container_width=True,
            hide_index=True,
        )


# --- Page: Incident Reconstruction -------------------------------------------
elif page == "Incident Reconstruction":
    page_header("CASE FILE", "Incident Reconstruction", "Turn a time range into a readable event timeline")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_time = st.number_input("Start time (s)", min_value=0.0, value=0.0, step=1.0)
    with col2:
        end_time = st.number_input("End time (s)", min_value=0.0, value=15.0, step=1.0)
    with col3:
        min_person_duration = st.slider("Min. person duration (s)", 0.0, 10.0, 1.0, 0.5,
                                         help="People visible for less than this, with no "
                                              "movement/meeting, are treated as background noise")

    if st.button("Generate Report", type="primary"):
        with st.spinner("Building timeline..."):
            data, err = api_get("/incident", params={
                "start_time": start_time, "end_time": end_time,
                "min_person_duration": min_person_duration,
            })
        if err:
            st.error(err)
        else:
            st.markdown(
                f'<span class="status-live">{data["num_events"]} EVENTS</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            for line in data["narrative"].split("\n"):
                if "--" in line:
                    ts, sentence = line.split("--", 1)
                    st.markdown(
                        f'<div style="display:flex;gap:12px;padding:4px 0;border-left:2px solid #2A2F37;padding-left:12px;margin-left:4px;">'
                        f'<span style="font-family:\'IBM Plex Mono\',monospace;color:#E8A33D;min-width:70px;">{ts.strip()}</span>'
                        f'<span>{sentence.strip()}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with st.expander("Raw structured events (JSON)"):
                st.json(data["events"])


# --- Page: Natural Language Search -------------------------------------------
elif page == "Natural Language Search":
    page_header("QUERY", "Natural Language Search", 'Describe who you\'re looking for — by typing or speaking')

    col_input, col_mic = st.columns([5, 1])
    with col_mic:
        voice_transcript = voice_input_button()
        if voice_transcript:
            st.session_state["nlsearch_query"] = voice_transcript
    with col_input:
        query = st.text_input(
            "Search query", key="nlsearch_query",
            placeholder="person wearing a red jacket", label_visibility="collapsed",
        )

    top_k = st.slider("Number of results", 1, 20, 5)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching... (can take 1-2 minutes on CPU-only hardware)"):
            data, err = api_get("/search/text", params={"query": query, "top_k": top_k}, timeout=180)
        if err:
            st.error(err)
            if "CLIP" in str(err) or "clip" in str(err):
                st.info("Run extract_clip_embeddings.py for each camera first.")
        else:
            for r in data["results"]:
                st.markdown(
                    f"**{r['camera_id'].upper()}** track {r['track_id']} &nbsp; "
                    + similarity_badge(r['similarity']) +
                    f" &nbsp; <span style='font-family:\"IBM Plex Mono\",monospace;color:#8B929E;'>"
                    f"{r['first_seen_timestamp']:.2f}s–{r['last_seen_timestamp']:.2f}s</span>",
                    unsafe_allow_html=True,
                )


# --- Page: Ask CatchCam ------------------------------------------------------------
elif page == "Ask CatchCam":
    page_header("CONVERSATION", "Ask CatchCam",
                "Ask about the footage in plain words, typed or spoken. Follow-ups about the same person work too")
    st.session_state.setdefault("qa_history", [])
    st.session_state.setdefault("qa_context", None)

    col_input, col_mic = st.columns([5, 1])
    with col_mic:
        spoken = voice_input_button()
        if spoken:
            st.session_state["qa_input"] = spoken
    with col_input:
        question = st.text_input("Your question", key="qa_input", label_visibility="collapsed",
                                 placeholder="Where did the person in the red jacket go?")

    col_ask, col_new = st.columns(2)
    with col_ask:
        ask_clicked = st.button("Ask", type="primary")
    with col_new:
        new_clicked = st.button("New conversation")

    if new_clicked:
        st.session_state["qa_history"], st.session_state["qa_context"] = [], None

    if (ask_clicked or spoken) and question.strip():
        params = {"question": question.strip()}
        ctx = st.session_state["qa_context"]
        if ctx and "plate" in ctx:
            params["context_plate"] = ctx["plate"]
        elif ctx:
            params["context_camera"], params["context_track"] = ctx["camera_id"], ctx["track_id"]
        with st.spinner("Looking through the footage..."):
            reply, err = api_get("/qa", params=params, timeout=60)
        if err:
            st.error(err)
            if "404" in str(err) or "Not Found" in str(err):
                st.info("The Q&A module did not load. Check http://localhost:8000/modules and the API terminal.")
        else:
            st.session_state["qa_history"].insert(0, reply)
            st.session_state["qa_context"] = reply.get("context")

    ctx = st.session_state["qa_context"]
    if ctx:
        who = f"vehicle {ctx['plate']}" if "plate" in ctx else f"{ctx['camera_id']} track {ctx['track_id']}"
        st.caption(f"Following up on {who}. Questions like \u201cwhat was he wearing?\u201d refer to this one. "
                   f"Press New conversation to start over.")

    with st.expander("Example questions"):
        examples, _ = api_get("/qa/examples")
        st.markdown("\n".join(f"- {e}" for e in (examples or {}).get("examples", [])) or "No examples available.")

    for i, item in enumerate(st.session_state["qa_history"]):
        st.markdown(
            f'<div style="margin-top:18px;color:#8B929E;">You asked</div>'
            f'<div style="font-weight:500;">{html.escape(item["question"])}</div>'
            f'<div style="margin-top:8px;border-left:3px solid #4FB8A8;padding:6px 12px;">{html.escape(item["answer"])}</div>',
            unsafe_allow_html=True,
        )
        note = f"Understood as: {item['intent']}" + (" (about the person from the previous answer)" if item.get("used_previous_subject") else "")
        st.caption(note)
        if i == 0:
            speak_button(item["speak"])
        with st.expander("Evidence behind this answer"):
            st.json(item.get("data", {}))


# --- Page: Clothing Search ------------------------------------------------------------
elif page == "Clothing Search":
    page_header("ATTRIBUTES", "Clothing Search",
                "Find people by what they wore. Colour of top and bottoms, voted over many frames")
    desc = st.text_input("Description", placeholder="person in a red jacket and blue jeans")
    col_a, col_b = st.columns(2)
    with col_a:
        search_clicked = st.button("Search People", type="primary")
    with col_b:
        verify_clicked = st.button("Check Re-ID Matches Against Clothing")

    if search_clicked:
        if not desc.strip():
            st.warning("Describe someone first, for example \u201cperson in a red jacket and blue jeans\u201d.")
        else:
            data, err = api_get("/attributes/search", params={"description": desc.strip()})
            if err:
                st.error(err)
                if "404" in str(err) and "Not Found" in str(err):
                    st.info("The attributes module did not load. Check http://localhost:8000/modules and the API terminal.")
            elif not data["results"]:
                st.info("Nobody matches. If you have not extracted clothing data yet, run "
                        "`python -m person_attrs.extract` for each camera first.")
            else:
                u = data["understood_as"]
                st.caption("Understood as: " + ", ".join(f"{k} = {v}" for k, v in u.items() if v))
                st.dataframe(
                    data["results"],
                    column_config={
                        "camera_id": "Camera", "track_id": "Track",
                        "score": st.column_config.NumberColumn("Match score", format="%.2f"),
                        "upper_color": "Top", "upper_conf": st.column_config.NumberColumn("Top confidence", format="%.2f"),
                        "lower_color": "Bottoms", "lower_conf": st.column_config.NumberColumn("Bottoms confidence", format="%.2f"),
                        "first_seen_timestamp": st.column_config.NumberColumn("First seen (s)", format="%.2f"),
                        "last_seen_timestamp": st.column_config.NumberColumn("Last seen (s)", format="%.2f"),
                        "num_samples": "Frames sampled",
                    },
                    use_container_width=True, hide_index=True,
                )

    if verify_clicked:
        data, err = api_get("/attributes/verify_matches")
        if err:
            st.error(err)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Clothing agrees", data["counts"]["consistent"])
            m2.metric("Clothing conflicts", data["counts"]["conflict"])
            m3.metric("Not enough data", data["counts"]["unknown"])
            if data["counts"]["conflict"]:
                st.warning(f"{data['counts']['conflict']} matched pair(s) wear completely different clothes. "
                           f"They may be two different people (or a change of clothes). Review these first.")
            rows = [{"Camera A": m["camera_a"], "Track A": m["track_a"], "Camera B": m["camera_b"], "Track B": m["track_b"],
                     "Re-ID similarity": m["similarity"], "Clothing": m["clothing"]["verdict"], "Why": m["clothing"]["reason"]}
                    for m in data["matches"]]
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("There are no cross-camera matches to check yet.")


# --- Page: Vehicles ------------------------------------------------------------
elif page == "Vehicles":
    page_header("VEHICLE INTELLIGENCE", "Vehicles", "Type, colour and number plate of tracked vehicles")

    col1, col2, col3 = st.columns(3)
    with col1:
        vtype = st.selectbox("Type", ["any", "car", "motorcycle", "bus", "truck"])
    with col2:
        vcolor = st.selectbox("Colour", ["any", "white", "black", "silver", "gray", "red", "blue",
                                          "green", "yellow", "orange", "brown", "purple"])
    with col3:
        vplate = st.text_input("Plate (full or partial)", placeholder="MH12AB1234")

    params = {}
    if vtype != "any":
        params["vehicle_type"] = vtype
    if vcolor != "any":
        params["color"] = vcolor
    if vplate.strip():
        params["plate"] = vplate.strip()

    col_find, col_follow = st.columns(2)
    with col_find:
        find_clicked = st.button("Search Vehicles", type="primary")
    with col_follow:
        follow_clicked = st.button("Follow Plate Across Cameras")

    if find_clicked:
        data, err = api_get("/vehicles", params=params)
        if err:
            st.error(err)
            if "404" in str(err) or "Not Found" in str(err):
                st.info("The vehicles module did not load. Check http://localhost:8000/modules and the API terminal.")
        elif not data["vehicles"]:
            st.info("No vehicles match. Run `python -m vehicle_intel.process_video` for each camera first.")
        else:
            st.markdown(f'<span class="status-live">{data["count"]} VEHICLES</span>', unsafe_allow_html=True)
            st.dataframe(
                data["vehicles"],
                column_config={
                    "camera_id": "Camera", "track_id": "Track", "vehicle_type": "Type", "color": "Colour",
                    "plate": "Plate", "plate_valid": "Valid format",
                    "plate_support": st.column_config.NumberColumn("Plate agreement", format="%.2f"),
                    "first_seen_timestamp": st.column_config.NumberColumn("First seen (s)", format="%.2f"),
                    "last_seen_timestamp": st.column_config.NumberColumn("Last seen (s)", format="%.2f"),
                    "num_frames": "Frames",
                },
                use_container_width=True, hide_index=True,
            )

    if follow_clicked:
        if not vplate.strip():
            st.warning("Enter a plate number to follow.")
        else:
            data, err = api_get("/vehicles/journey", params={"plate": vplate.strip()})
            if err:
                st.error(err)
            else:
                for i, s in enumerate(data["sightings"]):
                    st.markdown(
                        f"{'📍' if i == 0 else '↳'} **{s['camera_id'].upper()}** "
                        f"<span style='font-family:\"IBM Plex Mono\",monospace;'>{s['plate']} · {s['color']} {s['vehicle_type']} · "
                        f"{s['first_seen_timestamp']:.2f}s → {s['last_seen_timestamp']:.2f}s</span>",
                        unsafe_allow_html=True,
                    )
                for g in data["gaps"]:
                    st.warning(g["flag"])


# --- Page: Floor Plan ------------------------------------------------------------
elif page == "Floor Plan":
    page_header("MAP", "Floor Plan", "A person's journey drawn on the building plan, with unexplained gaps marked")

    cameras_data, _ = api_get("/cameras")
    camera_options = [c["camera_id"] for c in cameras_data["cameras"]] if cameras_data else []
    col1, col2, col3 = st.columns(3)
    with col1:
        fp_camera = st.selectbox("Camera", camera_options) if camera_options else st.text_input("Camera ID")
    with col2:
        fp_track = st.number_input("Track ID", min_value=0, step=1)
    with col3:
        fp_gap = st.number_input("Normal time between cameras (s)", min_value=1.0, value=15.0, step=1.0)

    col_show, col_cal = st.columns(2)
    with col_show:
        show_clicked = st.button("Show on Floor Plan", type="primary")
    with col_cal:
        cal_clicked = st.button("Check Camera Calibration")

    if show_clicked:
        params = {"camera_id": fp_camera, "track_id": fp_track, "max_normal_gap": fp_gap}
        png, err = api_get_bytes("/floorplan/render", params=params)
        if err:
            st.error(err)
            if "floorplan.json" in str(err):
                st.info("Create floorplan.json in your data folder (see floorplan/floorplan.example.json) with the plan size "
                        "and each camera's position. Calibrating a camera makes the path accurate; without it, the camera "
                        "is shown as a pin.")
        else:
            st.image(png, use_container_width=True)
            detail, err2 = api_get("/floorplan/journey", params=params)
            if detail:
                approx = [h["camera_id"] for h in detail["hops"] if h["mode"] != "projected"]
                if approx:
                    st.caption("Shown as pins (not calibrated or no usable path): " + ", ".join(a.upper() for a in approx))
                off = [h["camera_id"] for h in detail["hops"] if h["out_of_bounds_ratio"] > 0.3]
                if off:
                    st.warning("Much of the path on " + ", ".join(o.upper() for o in off) + " falls outside the plan. "
                               "That camera's calibration points are probably wrong. Use Check Camera Calibration.")
                for g in detail["gaps"]:
                    st.markdown(
                        f'<span class="status-down">{g["severity"].upper()}</span> &nbsp; '
                        f'{g["gap_seconds"]:.1f}s unseen between {g["from_camera"].upper()} and {g["to_camera"].upper()}',
                        unsafe_allow_html=True,
                    )

    if cal_clicked:
        data, err = api_get("/floorplan/config")
        if err:
            st.error(err)
        else:
            st.caption(f"Plan size: {data['width']} x {data['height']}")
            for cam, info in data["cameras"].items():
                if not info["calibrated"]:
                    st.markdown(f"**{cam.upper()}** &nbsp; not calibrated, shown as a pin at its position")
                elif info["ok"]:
                    st.markdown(f'**{cam.upper()}** &nbsp; <span class="status-live">● CALIBRATED</span> '
                                f'&nbsp; error {info["rms_error"]} (limit {info["tolerance"]})', unsafe_allow_html=True)
                else:
                    st.markdown(f'**{cam.upper()}** &nbsp; <span class="status-down">● RE-CHECK POINTS</span> '
                                f'&nbsp; error {info["rms_error"]} is above the limit {info["tolerance"]}', unsafe_allow_html=True)


# --- Page: Camera Health --------------------------------------------------------
elif page == "Camera Health":
    page_header("MONITORING", "Camera Health", "Which live feeds are still sending data")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        stale_after = st.slider("Mark a camera DOWN after this many seconds without an update", 5, 120, 10, 5)
    with col_b:
        auto_refresh = st.checkbox("Auto-refresh (5 s)")

    data, err = api_get("/camera_health", params={"stale_after": stale_after})
    if err:
        st.error(err)
        if "404" in str(err) or "Not Found" in str(err):
            st.info("The health module did not load. Check http://localhost:8000/modules and the API terminal.")
    elif data["status"] == "no_live_cameras":
        st.info("No live camera files found. Start `realtime_track.py` for a camera; it writes "
                "`outputs/<camera>_observations_live.json`, which this page watches.")
    else:
        total = len(data["cameras"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Cameras", total)
        m2.metric("Live", total - data["num_down"])
        m3.metric("Down", data["num_down"])
        if data["num_down"]:
            st.error(f"{data['num_down']} camera(s) stopped updating. Check the camera, cable or tracking process.")
        st.write("")
        for cam in data["cameras"]:
            pill = "status-down" if cam["is_down"] else "status-live"
            st.markdown(
                f'<div style="display:flex;gap:16px;align-items:center;padding:6px 0;border-bottom:1px solid #2A2F37;">'
                f'<b style="min-width:100px;">{cam["camera_id"].upper()}</b>'
                f'<span class="{pill}">● {"DOWN" if cam["is_down"] else "LIVE"}</span>'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;color:#8B929E;">'
                f'last update {cam["last_update_seconds_ago"]}s ago</span></div>',
                unsafe_allow_html=True,
            )

    if auto_refresh:
        time.sleep(5)
        st.rerun()
