"""
CatchCam — Incident Reconstruction Module
=============================================
Give it a time range, and it produces a chronological, human-readable
narrative of what happened across all cameras in that window -- turning
raw tracking data into something like:

    14:03 -- Person A enters through Camera 1
    14:07 -- Person A moves to Camera 3
    14:10 -- Person A meets Person B
    14:14 -- Both move toward Camera 5
    14:17 -- Person B exits the frame
    14:21 -- Person A exits the frame

This is the capstone feature: it reuses everything else you've built
(the Space-Time Graph for movement, relationship_tracking.py's output for
meetings) and adds one new step -- turning structured facts into natural
language using an LLM, which only narrates what's actually in the data
(it's told explicitly not to invent anything).

Two-stage design:
  1. build_incident_timeline() -- pure Python, no API needed. Groups track
     IDs into consistent "Person" identities using the match graph, then
     extracts entry/exit/movement/meeting events within your time range.
     This half is fully testable without any API key.
  2. narrate_timeline() -- sends the structured events to an LLM (Claude)
     to write the final natural-language report. Requires an API key.

Usage:
    python incident_reconstruction.py \
        --observations outputs/cam1_observations.json outputs/cam2_observations.json outputs/cam3_observations.json \
        --matches matches/cross_camera_matches.json \
        --relationships relationships/relationships.json \
        --start_time 0 --end_time 30 \
        --out_dir incidents

    # To skip the LLM step and just see the structured events (no API key needed):
    python incident_reconstruction.py ... --no_narrate

Requires: networkx (for grouping), anthropic (for narration -- pip install anthropic)
Set ANTHROPIC_API_KEY as an environment variable before running with narration.
"""
import argparse
import json
import os
from pathlib import Path

import networkx as nx

from build_graph import build_graph, node_id


def group_tracks_into_people(G: nx.DiGraph):
    """
    Uses the match graph's connected components to figure out which
    (camera, track_id) sightings belong to the same real person, and
    assigns each group a consistent, human-readable label.

    Returns: dict mapping node_id -> "Person A", "Person B", etc.
    """
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    # Sort components by their earliest timestamp, so "Person A" is
    # whoever appears first -- makes the narrative easier to follow
    components.sort(key=lambda comp: min(G.nodes[n]["first_seen_timestamp"] for n in comp))

    label_map = {}
    for i, comp in enumerate(components):
        label = f"Person {chr(65 + i)}" if i < 26 else f"Person {i+1}"
        for n in comp:
            label_map[n] = label
    return label_map


def build_incident_timeline(observation_paths, matches_path, relationships_path,
                             start_time: float, end_time: float):
    """
    Pure data-processing stage -- no API calls. Returns a list of
    structured events, sorted chronologically, restricted to the given
    [start_time, end_time] window (in the same relative-seconds units
    your track.py timestamps use).
    """
    G = build_graph(observation_paths, matches_path)
    labels = group_tracks_into_people(G)

    events = []

    # --- Entry / exit events, one pair per (camera, track) sighting ---
    for node, attrs in G.nodes(data=True):
        person = labels.get(node, "Unknown Person")
        if start_time <= attrs["first_seen_timestamp"] <= end_time:
            events.append({
                "type": "entry",
                "timestamp": attrs["first_seen_timestamp"],
                "person": person,
                "camera_id": attrs["camera_id"],
                "detail": f"{person} enters view on {attrs['camera_id']}",
            })
        if start_time <= attrs["last_seen_timestamp"] <= end_time:
            events.append({
                "type": "exit",
                "timestamp": attrs["last_seen_timestamp"],
                "person": person,
                "camera_id": attrs["camera_id"],
                "detail": f"{person} leaves view on {attrs['camera_id']}",
            })

    # --- Movement events: edges in the graph = a hop from one camera to another ---
    for src, dst, data in G.edges(data=True):
        src_attrs, dst_attrs = G.nodes[src], G.nodes[dst]
        person = labels.get(src, "Unknown Person")
        hop_time = dst_attrs["first_seen_timestamp"]
        if start_time <= hop_time <= end_time:
            events.append({
                "type": "movement",
                "timestamp": hop_time,
                "person": person,
                "camera_id": dst_attrs["camera_id"],
                "detail": f"{person} moves from {src_attrs['camera_id']} to {dst_attrs['camera_id']} "
                          f"(match confidence {data['similarity']:.2f})",
            })

    # --- Meeting events, from relationship_tracking.py's output ---
    if relationships_path and Path(relationships_path).exists():
        with open(relationships_path) as f:
            relationships = json.load(f)
        for rel in relationships:
            node_a = node_id(rel["camera_id"], rel["track_a"])
            node_b = node_id(rel["camera_id"], rel["track_b"])
            person_a = labels.get(node_a, f"track {rel['track_a']}")
            person_b = labels.get(node_b, f"track {rel['track_b']}")
            if person_a == person_b:
                continue  # same real person somehow flagged with themself -- skip
            if start_time <= rel["start_timestamp"] <= end_time:
                events.append({
                    "type": "meeting",
                    "timestamp": rel["start_timestamp"],
                    "person": f"{person_a} & {person_b}",
                    "camera_id": rel["camera_id"],
                    "detail": f"{person_a} and {person_b} are together on {rel['camera_id']} "
                              f"for {rel['duration_seconds']:.1f}s",
                })

    events.sort(key=lambda e: e["timestamp"])

    # De-duplicate near-identical events (same type/person/camera within a
    # fraction of a second of each other) -- these come from genuine
    # duplicate matches in noisy data, and just confuse the LLM without
    # adding real information.
    deduped = []
    seen = set()
    for e in events:
        key = (e["type"], e["person"], e["camera_id"], round(e["timestamp"], 1))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    return deduped


def narrate_timeline_template(events, start_time, end_time, min_person_duration: float = 1.0):
    """
    Deterministic, template-based narration -- no LLM, no hallucination risk.
    Converts each event into a fixed sentence pattern directly from the
    structured data. Guaranteed 100% grounded, since there's no free-form
    generation happening at all.

    min_person_duration: people whose total visible time is below this AND
    who have no movement/meeting events are treated as background noise
    (e.g. a single stray frame) and dropped from the narrative -- their
    raw data still exists in the saved events JSON, just not narrated.
    """
    # Figure out each person's total visible duration and whether they had
    # any "significant" event (movement or meeting), to decide who's worth
    # narrating vs. who's just noise.
    person_activity = {}
    for e in events:
        for person in e["person"].split(" & "):  # meeting events have "A & B"
            if person not in person_activity:
                person_activity[person] = {"duration": 0.0, "significant": False}
            if e["type"] in ("movement", "meeting"):
                person_activity[person]["significant"] = True

    entries = {e["person"]: e["timestamp"] for e in events if e["type"] == "entry"}
    exits = {e["person"]: e["timestamp"] for e in events if e["type"] == "exit"}
    for person in person_activity:
        if person in entries and person in exits:
            person_activity[person]["duration"] = exits[person] - entries[person]

    def is_worth_narrating(person):
        act = person_activity.get(person)
        if act is None:
            return True  # meeting-only labels like "A & B" -- always keep
        return act["significant"] or act["duration"] >= min_person_duration

    lines = []
    for e in events:
        people_in_event = e["person"].split(" & ")
        if not any(is_worth_narrating(p) for p in people_in_event):
            continue

        if e["type"] == "entry":
            sentence = f"{e['person']} enters view on {e['camera_id']}."
        elif e["type"] == "exit":
            sentence = f"{e['person']} leaves view on {e['camera_id']}."
        elif e["type"] == "movement":
            # detail already has the full "moves from X to Y" phrasing
            sentence = e["detail"][0].upper() + e["detail"][1:] + "."
        elif e["type"] == "meeting":
            sentence = e["detail"][0].upper() + e["detail"][1:] + "."
        else:
            continue

        lines.append(f"[{e['timestamp']:.2f}s] -- {sentence}")

    if not lines:
        return "No significant events found in this time range."

    return "\n".join(lines)


def narrate_timeline_anthropic(events, start_time, end_time, api_key=None, model="claude-sonnet-4-6"):
    """
    Sends the structured events to Claude (via the Anthropic API) to write
    a natural-language incident report. Requires an ANTHROPIC_API_KEY with
    available credits.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)  # reads ANTHROPIC_API_KEY env var if not passed

    prompt = build_narration_prompt(events, start_time, end_time)

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def narrate_timeline_ollama(events, start_time, end_time,
                             model="llama3.2:1b", host="http://localhost:11434"):
    """
    Sends the structured events to a locally running Ollama model instead
    of a cloud API -- fully free, fully offline, no API key needed.

    Requires Ollama installed and running (https://ollama.com), with the
    model already pulled, e.g.: ollama pull llama3.2:1b

    Recommended models for a CPU-only laptop (fastest to slowest, roughly):
      llama3.2:1b   -- fastest, good enough for this structured task
      qwen2.5:1.5b  -- similar size, sometimes better instruction-following
      phi3:mini     -- a bit larger (~3.8B) but still CPU-friendly, higher quality
    """
    import json as _json
    import urllib.request
    import urllib.error

    prompt = build_narration_prompt(events, start_time, end_time)

    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            result = _json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Is Ollama installed and running? "
            f"(Try: ollama serve, or just open the Ollama app). "
            f"Original error: {e}"
        )


def build_narration_prompt(events, start_time, end_time):
    """Shared prompt builder used by both the Anthropic and Ollama paths."""
    events_text = "\n".join(
        f"- [{e['timestamp']:.2f}s] ({e['type']}) {e['detail']}"
        for e in events
    )

    return f"""You are writing an incident reconstruction report for a CCTV surveillance system.
Below is a list of factual, timestamped events extracted automatically from multi-camera
tracking data, covering the time window {start_time:.2f}s to {end_time:.2f}s.

RULES:
- Use ONLY the events listed below. Do not invent people, cameras, locations, or events.
- Convert each event into one clear, natural sentence.
- Keep the original person labels (Person A, Person B, etc.) exactly as given.
- Order strictly by timestamp.
- Format each line as: [timestamp] -- sentence
- Do not add commentary, speculation, or a summary paragraph -- just the timeline.

EVENTS:
{events_text}

Write the incident timeline now:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", nargs="+", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--relationships", default=None)
    parser.add_argument("--start_time", type=float, required=True)
    parser.add_argument("--end_time", type=float, required=True)
    parser.add_argument("--out_dir", default="incidents")
    parser.add_argument("--no_narrate", action="store_true",
                         help="Skip the LLM step, just output structured events (no API key needed)")
    parser.add_argument("--provider", choices=["template", "anthropic", "ollama"], default="template",
                         help="'template' = deterministic, no AI, always accurate (default). "
                              "'ollama' = free local LLM, more natural prose but can drift from facts. "
                              "'anthropic' = cloud API, requires a paid key, best quality.")
    parser.add_argument("--model", default=None,
                         help="Model name. Defaults: claude-sonnet-4-6 for anthropic, "
                              "llama3.2:1b for ollama. Not used for 'template'.")
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    parser.add_argument("--min_person_duration", type=float, default=1.0,
                         help="[template mode] Minimum seconds a person must be visible "
                              "(with no movement/meeting) to be included in the narrative, "
                              "filtering out likely noise. Set to 0 to include everyone.")
    args = parser.parse_args()

    events = build_incident_timeline(
        args.observations, args.matches, args.relationships,
        args.start_time, args.end_time,
    )

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    events_file = out_path / f"events_{args.start_time:.0f}_{args.end_time:.0f}.json"
    with open(events_file, "w") as f:
        json.dump(events, f, indent=2)

    print(f"[CatchCam] Found {len(events)} events between {args.start_time}s and {args.end_time}s")
    print(f"[CatchCam] Structured events saved -> {events_file}\n")

    for e in events:
        print(f"  [{e['timestamp']:.2f}s] ({e['type']}) {e['detail']}")

    if not args.no_narrate:
        if not events:
            print("\n[CatchCam] No events in this range -- skipping narration.")
            return
        print("\n[CatchCam] Generating incident report...")
        if args.provider == "template":
            narrative = narrate_timeline_template(events, args.start_time, args.end_time,
                                                    min_person_duration=args.min_person_duration)
        elif args.provider == "ollama":
            model = args.model or "llama3.2:1b"
            narrative = narrate_timeline_ollama(events, args.start_time, args.end_time,
                                                 model=model, host=args.ollama_host)
        else:
            model = args.model or "claude-sonnet-4-6"
            narrative = narrate_timeline_anthropic(events, args.start_time, args.end_time, model=model)

        report_file = out_path / f"report_{args.start_time:.0f}_{args.end_time:.0f}.txt"
        with open(report_file, "w") as f:
            f.write(narrative)

        print("\n" + "=" * 60)
        print("INCIDENT RECONSTRUCTION REPORT")
        print("=" * 60)
        print(narrative)
        print("=" * 60)
        print(f"\n[CatchCam] Report saved -> {report_file}")


if __name__ == "__main__":
    main()
