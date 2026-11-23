"""
Conversational Q&A over CatchCam's own data.  "Where did the person in the red jacket go?"  ->  a spoken-style answer.

Design: a deterministic parser (intent + subject) calls the SAME functions the rest of the system uses (Space-Time
Graph, coverage gaps, relationships, clothing attributes, vehicles, camera health) and writes the answer from the
results. Nothing is generated freely, so it cannot invent a camera, time or person -- every sentence comes from data.
(An LLM can be bolted on later purely to rephrase `answer`; the facts would still come from here.)

Needs the project's own build_graph.py and missing_camera.py on the import path.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import networkx as nx

from build_graph import build_graph, node_id, reconstruct_journey, reconstruct_journey_backward
from missing_camera import check_live_camera_health, find_coverage_gaps, get_full_journey
from person_attrs.core import matches_all, parse_description, search_people
from vehicle_intel.core import plate_distance

EXAMPLES = [
    "Where did the person in the red jacket go?",
    "Where did track 5 on cam1 come from?",
    "Show the journey of the person in a blue shirt and black pants",
    "Who was with track 3 on cam2?",
    "How many people were seen between 10 and 60 seconds?",
    "Are there any blind spots in the journey of track 1 on cam1?",
    "What was track 4 on cam2 wearing?",
    "Which cameras are down?",
    "Where did the car MH12AB1234 go?",
    "How many red cars were seen?",
]

VEHICLE_WORDS = {"car": "car", "cars": "car", "bike": "motorcycle", "motorcycle": "motorcycle", "motorbike": "motorcycle",
                 "bus": "bus", "truck": "truck", "lorry": "truck", "van": "car", "vehicle": None, "vehicles": None}


# ----------------------------------------------------------------------------- data access
class Store:
    def __init__(self, data_dir="."):
        self.data_dir = Path(data_dir)
        self.outputs = self.data_dir / "outputs"
        self.matches = self.data_dir / "matches" / "cross_camera_matches.json"
        self.relationships_file = self.data_dir / "relationships" / "relationships.json"
        self._G = None

    def _load(self, files):
        rows = []
        for f in files:
            rows += json.loads(f.read_text(encoding="utf-8-sig"))
        return rows

    def obs_files(self):
        return sorted(self.outputs.glob("*_observations.json")) if self.outputs.exists() else []

    def graph(self) -> Optional[nx.DiGraph]:
        if self._G is None and self.obs_files() and self.matches.exists():
            self._G = build_graph([str(f) for f in self.obs_files()], str(self.matches))
        return self._G

    def attributes(self):
        return self._load(sorted(self.outputs.glob("*_attributes.json"))) if self.outputs.exists() else []

    def vehicles(self):
        return self._load(sorted(self.outputs.glob("*_vehicles.json"))) if self.outputs.exists() else []

    def relationships(self):
        return json.loads(self.relationships_file.read_text(encoding="utf-8-sig")) if self.relationships_file.exists() else []

    def camera_ids(self):
        G = self.graph()
        return sorted({d["camera_id"] for _, d in G.nodes(data=True)}) if G else []


# ----------------------------------------------------------------------------- subject resolution
@dataclass
class Subject:
    kind: str = "none"            # track | plate | vehicles | none
    camera_id: Optional[str] = None
    track_id: Optional[int] = None
    plate: Optional[str] = None
    vehicle_rows: Optional[list] = None
    label: str = ""
    note: str = ""                # e.g. "3 people match; using the best match"
    verified: bool = False        # True once the sighting/vehicle was actually found in the data (only then do we remember it)


PLATE_RE = re.compile(r"\b([a-z]{2})[\s\-]?(\d{1,2})[\s\-]?([a-z]{1,3})[\s\-]?(\d{4})\b")


def _camera_from_text(q: str, known: List[str]) -> Optional[str]:
    for cid in known:                                            # exact id mentioned, e.g. "cam2"
        if re.search(rf"\b{re.escape(cid.lower())}\b", q):
            return cid
    m = re.search(r"\b(?:camera|cam)[\s_\-]*(\d+)\b", q)
    if m:
        n = m.group(1)
        for cid in known:
            if re.sub(r"\D", "", cid) == n:
                return cid
        return f"cam{n}"
    return None


def resolve_subject(q: str, store: Store) -> Subject:
    G = store.graph()
    known = store.camera_ids()

    m = PLATE_RE.search(q)
    if m:
        plate = "".join(m.groups()).upper()
        rows = [r for r in store.vehicles() if r.get("plate") and plate_distance(plate, r["plate"]) <= (1 if len(plate) >= 8 else 0)]
        return Subject("plate", plate=plate, vehicle_rows=rows, label=f"vehicle {plate}")

    tm = re.search(r"(?:\b(?:track|person|subject|people|id)|#)[\s_]*(?:id|number|no\.?)?[\s#:]*(\d+)\b", q)
    if tm:
        tid, cam = int(tm.group(1)), _camera_from_text(q, known)
        if cam is None and G:
            holders = sorted((d["camera_id"] for n, d in G.nodes(data=True) if d["track_id"] == tid))
            cam = holders[0] if holders else None
            if len(holders) > 1:
                return Subject("track", cam, tid, label=f"{cam} track {tid}",
                               note=f"Track {tid} exists on several cameras ({', '.join(holders)}); using {cam}.")
        return Subject("track", cam, tid, label=f"{cam} track {tid}")

    words = set(re.findall(r"[a-z]+", q))
    vtype_words = [w for w in words if w in VEHICLE_WORDS]
    want = parse_description(q)
    if vtype_words:                                              # "the red car", "white bus"
        vtype = next((VEHICLE_WORDS[w] for w in vtype_words if VEHICLE_WORDS[w]), None)
        colour = want.get("any") or want.get("upper") or want.get("lower")
        rows = [r for r in store.vehicles() if (not vtype or r["vehicle_type"] == vtype) and (not colour or r["color"] == colour)]
        label = " ".join(x for x in (colour, vtype or "vehicle") if x)
        return Subject("vehicles", vehicle_rows=rows, label=f"{label}")
    if any(want.values()):
        people = [r for r in search_people(store.attributes(), q, limit=20) if matches_all(r, want)]   # all parts must match
        if not people:
            return Subject("none", label="description", note="")
        best = people[0]
        desc = " and ".join(x for x in (f"{want['upper']} top" if want["upper"] else "", f"{want['lower']} bottoms" if want["lower"] else "",
                                          f"{want['any']} clothing" if want["any"] else "") if x)
        note = (f"{len(people)} people match; answering for the best match." if len(people) > 1 else "")
        return Subject("track", best["camera_id"], best["track_id"], label=f"the person in {desc} ({best['camera_id']} track {best['track_id']})", note=note)
    return Subject("none")


# ----------------------------------------------------------------------------- conversation memory
def context_of(subject: Optional[Subject]) -> Optional[dict]:
    """What the dashboard should send back with the NEXT question so 'he', 'that car', 'them' resolve."""
    if not subject or not subject.verified:
        return None
    if subject.kind == "track":
        return {"camera_id": subject.camera_id, "track_id": subject.track_id}
    if subject.kind == "plate":
        return {"plate": subject.plate}
    return None


def subject_from_context(context: Optional[dict], store: Store) -> Optional[Subject]:
    """Rebuild the previous subject. Returns None if the context is empty or no longer matches the data."""
    if not context:
        return None
    if context.get("plate"):
        plate = str(context["plate"]).upper()
        rows = [r for r in store.vehicles() if r.get("plate") and plate_distance(plate, r["plate"]) <= (1 if len(plate) >= 8 else 0)]
        return Subject("plate", plate=plate, vehicle_rows=rows, label=f"vehicle {plate}", verified=bool(rows))
    cam, tid = context.get("camera_id"), context.get("track_id")
    if cam is None or tid is None:
        return None
    try:
        tid = int(tid)
    except (TypeError, ValueError):
        return None
    G = store.graph()
    if G is None or node_id(cam, tid) not in G.nodes:
        return None
    return Subject("track", cam, tid, label=f"{cam} track {tid}", verified=True)


# ----------------------------------------------------------------------------- formatting helpers
def _t(x) -> str:
    """A moment on the shared clock: 12.5s for short clips, 12:38:56 once cameras have real start times (an hour or more)."""
    if x >= 3600:
        h, rem = divmod(int(x), 3600)
        return f"{h}:{rem // 60:02d}:{rem % 60:02d}"
    return f"{x:.1f}s"


def _hop(h) -> str:
    return f"{h['camera_id']} (track {h['track_id']}) from {_t(h['first_seen_timestamp'])} to {_t(h['last_seen_timestamp'])}"


def _route(j) -> str:
    return " → ".join(h["camera_id"] for h in j)


def speakable(text: str) -> str:
    t = text.replace("→", "then").replace("(", ", ").replace(")", ", ")
    t = re.sub(r"\bcam(\d+)\b", r"camera \1", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+,", ",", t)                      # no space before a comma
    t = re.sub(r",(\s*,)+", ",", t)                  # collapse ",,"
    t = re.sub(r",\s*([.;:!?])", r"\1", t)          # ", ." -> "."
    return re.sub(r"[,\s]+$", "", t)                # no dangling comma at the end


def _gap_sentences(journey, max_gap) -> List[str]:
    return [f"Warning: unaccounted for {g['gap_seconds']:.1f}s between {g['from_camera']} and {g['to_camera']} "
            f"(last seen {_t(g['last_seen'])}, next seen {_t(g['next_seen'])})." for g in find_coverage_gaps(journey, max_gap)]


# ----------------------------------------------------------------------------- intent detection
def detect_intent(q: str) -> str:
    if re.search(r"\b(camera|cameras|cam)\b.*\b(down|offline|working|healthy|health|status|alive|disconnected|failed)\b", q) \
            or re.search(r"\b(down|offline)\b.*\bcameras?\b", q):
        return "camera_health"
    if re.search(r"\b(gap|gaps|blind spot|blind spots|unaccounted|disappear\w*|vanish\w*|missing time)\b", q):
        return "gaps"
    if re.search(r"\b(wearing|wore|dressed|clothes|clothing|outfit|look like|colou?r of)\b", q):
        return "wearing"
    if re.search(r"\bwho\b.*\b(with|meet|met|together|near|accompan\w*)\b|\bwhich people\b.*\b(with|together)\b", q):
        return "with_whom"
    if re.search(r"\bhow many cameras\b|\b(which|what) cameras\b|\bcameras? (did|has|was)\b", q):
        return "cameras"
    if re.search(r"\bhow many\b", q):
        return "count"
    if re.search(r"\b(come|came|coming|arrive\w*|enter\w*|originat\w*|start\w*)\b.*\bfrom\b|\bwhere\b.*\bfrom\b|\borigin\b", q):
        return "came_from"
    if re.search(r"\bwhere\b.*\b(go|went|going|head\w*|end\w*|exit\w*|leave|left|now)\b", q):
        return "went_to"
    if re.search(r"\b(journey|path|route|trace|trail|movements?|track him|follow)\b", q):
        return "journey"
    if re.search(r"\bwhen\b|\bwhat time\b|\bfirst seen\b|\blast seen\b", q):
        return "when"
    return "unknown"


# ----------------------------------------------------------------------------- answering
def ask(question: str, store: Optional[Store] = None, max_normal_gap: float = 15.0, context: Optional[dict] = None) -> dict:
    """
    context: the `context` returned by the previous answer ({"camera_id","track_id"} or {"plate"}). If the question
    names no new person/vehicle ("what was he wearing?", "and where did he go next?") it is about that subject.
    """
    store = store or Store()
    q = re.sub(r"\s+", " ", (question or "").lower().strip())
    intent = detect_intent(q)
    base = {"question": question, "intent": intent}
    followed = {"used": False}
    remembered = subject_from_context(context, store)       # None if there is no context OR it no longer matches the data
    context = context_of(remembered)                        # so a stale context is dropped, never passed along

    def reply(text, subject=None, data=None):
        full = text + (f" ({subject.note})" if subject and subject.note else "")
        spoken = text + (" " + subject.note if subject and subject.note else "")
        # remember the subject for the next question; if this question had none, keep the old one
        new_ctx = context_of(subject) or (context if (context and not (subject and subject.kind in ("track", "plate"))) else None)
        return {**base, "answer": full, "speak": speakable(spoken), "subject": subject.label if subject else "", "data": data or {},
                "context": new_ctx, "used_previous_subject": followed["used"]}

    if not q:
        return reply("Ask me something about the footage, for example: " + EXAMPLES[0])

    # ---- no subject needed
    if intent == "camera_health":
        res = check_live_camera_health(str(store.outputs), 10.0)
        if res["status"] == "no_live_cameras":
            return reply("There are no live camera feeds to check right now.", data=res)
        down = [c["camera_id"] for c in res["cameras"] if c["status"].startswith("DOWN")]
        if down:
            return reply(f"{len(down)} camera{'s are' if len(down) > 1 else ' is'} down: {', '.join(down)}. "
                         f"The others are live.", data=res)
        return reply(f"All {len(res['cameras'])} live cameras are reporting.", data=res)

    G = store.graph()

    if intent == "count":
        return _answer_count(q, store, G, reply)
    if intent == "unknown":
        # a bare subject ("person in blue shirt", "track 5 on cam1", a plate) with no question word means "find them":
        # treat it as a journey request instead of refusing
        guess = resolve_subject(q, store)
        if guess.kind != "none" or guess.label == "description":
            intent = base["intent"] = "journey"
    if intent == "unknown":
        return reply("I didn't understand that. Try asking things like: " + " / ".join(EXAMPLES[:4]))

    subj = resolve_subject(q, store)
    if subj.kind == "none" and subj.label != "description":
        if remembered is not None:                                  # "what was he wearing?" -> the person from before
            subj, followed["used"] = remembered, True
    if subj.kind == "none" and subj.label == "description":
        return reply("Nobody in the data matches that clothing description"
                     + ("" if store.attributes() else " (no clothing data has been extracted yet: run person_attrs.extract)") + ".", subj)
    if subj.kind == "none":
        return reply("I couldn't tell who or what you mean. Name a track (\"track 5 on cam1\"), a plate, or a clothing "
                     "description (\"the person in a red jacket\").", subj)

    # ---- vehicles
    if subj.kind in ("plate", "vehicles"):
        subj.verified = bool(subj.vehicle_rows) and subj.kind == "plate"
        return _answer_vehicle(intent, subj, reply)

    # ---- people (a track)
    if G is None:
        return reply("There is no tracking and matching data loaded yet, so I can't follow anyone.", subj)
    if not subj.camera_id or node_id(subj.camera_id, subj.track_id) not in G.nodes:
        return reply(f"I can't find a sighting for {subj.label or 'that track'} in the data.", subj)
    cam, tid = subj.camera_id, subj.track_id
    subj.verified = True
    node_attrs = G.nodes[node_id(cam, tid)]

    if intent == "wearing":
        rows = [r for r in store.attributes() if r["camera_id"] == cam and r["track_id"] == tid]
        if not rows:
            return reply(f"I have no clothing data for {cam} track {tid}. Run person_attrs.extract for {cam} first.", subj)
        r = rows[0]
        return reply(f"{cam} track {tid} was wearing a {r['upper_color']} top (confidence {r['upper_conf']:.0%}) and "
                     f"{r['lower_color']} bottoms (confidence {r['lower_conf']:.0%}).", subj, r)

    if intent == "with_whom":
        rel = [r for r in store.relationships() if r["camera_id"] == cam and tid in (r["track_a"], r["track_b"])]
        if not rel:
            return reply(f"{cam} track {tid} was not seen together with anyone for a sustained period.", subj)
        attrs = {(r["camera_id"], r["track_id"]): r for r in store.attributes()}
        parts = []
        for r in sorted(rel, key=lambda r: r["start_timestamp"]):
            other = r["track_b"] if r["track_a"] == tid else r["track_a"]
            a = attrs.get((cam, other))
            clothes = f" (wearing {a['upper_color']} top, {a['lower_color']} bottoms)" if a else ""
            parts.append(f"track {other}{clothes} from {_t(r['start_timestamp'])} to {_t(r['end_timestamp'])} "
                         f"({r['duration_seconds']:.1f}s together)")
        return reply(f"On {cam}, track {tid} was with " + "; ".join(parts) + ".", subj, {"relationships": rel})

    full = get_full_journey(G, cam, tid)
    if intent == "went_to":
        j = reconstruct_journey(G, cam, tid)
        if len(j) == 1:
            return reply(f"After {cam} track {tid} ({_t(j[0]['last_seen_timestamp'])}), no later sighting on another camera "
                         f"was matched, so I don't know where they went next.", subj, {"journey": j})
        text = f"Route: {_route(j)}. " + " ".join(f"On {h['camera_id']}: {_t(h['first_seen_timestamp'])} to {_t(h['last_seen_timestamp'])}." for h in j[1:])
        return reply(" ".join([text] + _gap_sentences(j, max_normal_gap)), subj, {"journey": j})
    if intent == "came_from":
        j = reconstruct_journey_backward(G, cam, tid)
        if len(j) == 1:
            return reply(f"No earlier sighting on another camera was matched before {cam} track {tid} "
                         f"(first seen at {_t(j[0]['first_seen_timestamp'])}), so this looks like where they first appeared.", subj, {"journey": j})
        return reply(f"They came from {j[0]['camera_id']} (seen {_t(j[0]['first_seen_timestamp'])} to {_t(j[0]['last_seen_timestamp'])}), "
                     f"route {_route(j)}. " + " ".join(_gap_sentences(j, max_normal_gap)), subj, {"journey": j})
    if intent == "journey":
        text = f"Full journey: {_route(full)}. " + " ".join(f"{_hop(h)}." for h in full)
        return reply(" ".join([text] + _gap_sentences(full, max_normal_gap)), subj, {"journey": full})
    if intent == "cameras":
        cams = list(dict.fromkeys(h["camera_id"] for h in full))
        return reply(f"They appeared on {len(cams)} camera{'s' if len(cams) != 1 else ''}: {', '.join(cams)}.", subj, {"journey": full})
    if intent == "when":
        return reply(f"First seen at {_t(full[0]['first_seen_timestamp'])} on {full[0]['camera_id']}; last seen at "
                     f"{_t(full[-1]['last_seen_timestamp'])} on {full[-1]['camera_id']}. This sighting ({cam} track {tid}) "
                     f"ran {_t(node_attrs['first_seen_timestamp'])} to {_t(node_attrs['last_seen_timestamp'])}.", subj, {"journey": full})
    if intent == "gaps":
        gaps = find_coverage_gaps(full, max_normal_gap)
        if not gaps:
            return reply(f"No unexplained gaps: every hop on the route {_route(full)} is within {max_normal_gap:.0f} seconds.", subj, {"journey": full})
        return reply(f"Route {_route(full)}. " + " ".join(_gap_sentences(full, max_normal_gap)), subj, {"journey": full, "gaps": gaps})
    return reply("I understood the question but can't answer that kind yet.", subj)


def _answer_count(q, store, G, reply):
    words = set(re.findall(r"[a-z]+", q))
    if words & set(VEHICLE_WORDS):
        subj = resolve_subject(q, store)
        rows = subj.vehicle_rows or []
        return reply(f"I counted {len(rows)} {subj.label}{'s' if len(rows) != 1 and not subj.label.endswith('s') else ''} in the data.", subj, {"vehicles": rows})
    if G is None:
        return reply("There is no tracking data loaded yet, so I can't count people.")
    cam = _camera_from_text(q, store.camera_ids())
    t0, t1 = None, None
    m = re.search(r"(?:between|from)\s+(\d+(?:\.\d+)?)\s*(?:s|sec\w*)?\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)", q)
    if m:
        t0, t1 = float(m.group(1)), float(m.group(2))
    nodes = [n for n, d in G.nodes(data=True)
             if (cam is None or d["camera_id"] == cam)
             and (t0 is None or (d["last_seen_timestamp"] >= t0 and d["first_seen_timestamp"] <= t1))]
    people = nx.number_weakly_connected_components(G.subgraph(nodes)) if nodes else 0
    where = f" on {cam}" if cam else ""
    when = f" between {t0:g}s and {t1:g}s" if t0 is not None else ""
    if not nodes:
        return reply(f"No one was seen{where}{when}.")
    return reply(f"{len(nodes)} sighting{'s' if len(nodes) != 1 else ''}{where}{when}, which is about {people} distinct "
                 f"{'person' if people == 1 else 'people'} after linking the same person across cameras.",
                 data={"sightings": len(nodes), "people": people})


def _answer_vehicle(intent, subj, reply):
    rows = sorted(subj.vehicle_rows or [], key=lambda r: r["first_seen_timestamp"] or 0)
    if not rows:
        return reply(f"I have no sightings of {subj.label}.", subj)
    desc = lambda r: f"{r['camera_id']} ({_t(r['first_seen_timestamp'])} to {_t(r['last_seen_timestamp'])}, {r['color']} {r['vehicle_type']}{', plate ' + r['plate'] if r['plate'] else ''})"
    route = " → ".join(dict.fromkeys(r["camera_id"] for r in rows))
    if intent in ("went_to", "journey", "cameras", "gaps", "when"):
        gaps = find_coverage_gaps(rows, 15.0)
        text = f"{subj.label.capitalize()} was seen {len(rows)} time{'s' if len(rows) != 1 else ''}: " + "; ".join(desc(r) for r in rows) + "."
        if intent == "went_to":
            text = f"The last sighting of {subj.label} was on {rows[-1]['camera_id']} at {_t(rows[-1]['last_seen_timestamp'])}. Route: {route}."
        return reply(" ".join([text] + _gap_sentences(rows, 15.0)), subj, {"sightings": rows, "gaps": gaps})
    if intent == "came_from":
        return reply(f"{subj.label.capitalize()} was first seen on {rows[0]['camera_id']} at {_t(rows[0]['first_seen_timestamp'])}.", subj, {"sightings": rows})
    return reply(f"{subj.label.capitalize()}: " + "; ".join(desc(r) for r in rows) + ".", subj, {"sightings": rows})
