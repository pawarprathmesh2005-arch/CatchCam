"""
CatchCam — incident report (PDF) for one person's journey.

Built from the same data the Investigate screen shows (webapp.core.subject_bundle), so the report and the screen never disagree.
Pictures come from the stored footage. Needs: reportlab (pip install reportlab).

Honesty rules baked into the document: it says the matches are automatic and probabilistic, shows the appearance-match score
and the clothing check for every link, marks gaps as "unaccounted" (not "out of sight and up to no good"), and lists what it does
not know. It never shows a face-recognition result because CatchCam does none.
"""
from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from webapp import core as web

INK, MUTED, LINE, ACCENT, BAD, WARN = colors.HexColor("#1b2430"), colors.HexColor("#5d6b7e"), colors.HexColor("#d5dbe3"), colors.HexColor("#e0a800"), colors.HexColor("#c0392b"), colors.HexColor("#d9822b")
PAGE_W = A4[0] - 4 * cm


def _t(x) -> str:
    if x is None:
        return "-"
    m, s = divmod(float(x), 60)
    return f"{int(m)}:{s:04.1f}"


def _styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK)
    return {
        "base": base,
        "title": ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=2),
        "sub": ParagraphStyle("sub", parent=base, textColor=MUTED, fontSize=10),
        "h": ParagraphStyle("h", parent=base, fontName="Helvetica-Bold", fontSize=12.5, leading=16, spaceBefore=14, spaceAfter=6),
        "small": ParagraphStyle("small", parent=base, fontSize=8, leading=10.5, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.5, leading=11),
        "cellb": ParagraphStyle("cellb", parent=base, fontSize=8.5, leading=11, fontName="Helvetica-Bold"),
    }


def _clothing_text(c: Optional[dict]) -> str:
    if not c or not (c.get("upper") or c.get("lower")):
        return "no clothing data"
    return f"{c.get('upper') or '?'} top, {c.get('lower') or '?'} bottoms"


def _timeline(hops: List[dict], gaps: List[dict]) -> Drawing:
    cams = []
    for h in hops:
        if h["camera_id"] not in cams:
            cams.append(h["camera_id"])
    lane_h, left, right_pad = 22, 52, 10
    height = lane_h * len(cams) + 34
    d = Drawing(PAGE_W, height)
    t0, t1 = min(h["start"] for h in hops), max(h["end"] for h in hops)
    span = max(t1 - t0, 1e-6)
    w = PAGE_W - left - right_pad
    x = lambda t: left + (t - t0) / span * w                                   # noqa: E731
    y = lambda cam: height - 22 - cams.index(cam) * lane_h                     # noqa: E731
    for cam in cams:
        d.add(Rect(left, y(cam) - 4, w, 14, fillColor=colors.HexColor("#f3f5f8"), strokeColor=None))
        d.add(String(0, y(cam), cam, fontName="Helvetica", fontSize=8.5, fillColor=INK))
    for i in range(0, 5):
        tt = t0 + span * i / 4
        d.add(Line(x(tt), 6, x(tt), height - 12, strokeColor=LINE, strokeWidth=0.4))
        d.add(String(x(tt), 0, _t(tt), fontName="Helvetica", fontSize=7, fillColor=MUTED, textAnchor="middle"))
    for a, b in zip(hops, hops[1:]):
        gap = any(g["from_camera"] == a["camera_id"] and g["to_camera"] == b["camera_id"] and g["next_seen"] == b["start"] for g in gaps)
        ln = Line(x(a["end"]), y(a["camera_id"]) + 3, x(b["start"]), y(b["camera_id"]) + 3, strokeColor=BAD if gap else MUTED, strokeWidth=1.2)
        if gap:
            ln.strokeDashArray = [3, 2]
        d.add(ln)
    for h in hops:
        d.add(Rect(x(h["start"]), y(h["camera_id"]) - 2, max(x(h["end"]) - x(h["start"]), 2.5), 10, fillColor=ACCENT, strokeColor=INK, strokeWidth=0.5))
    for g in gaps:
        mid = (x(g["last_seen"]) + x(g["next_seen"])) / 2
        d.add(String(mid, height - 10, f"{g['gap_seconds']:.0f}s unaccounted", fontName="Helvetica-Bold", fontSize=7.5, fillColor=BAD, textAnchor="middle"))
    return d


def _table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
    t.setStyle(TableStyle(style))
    return t


def _floor_plan_png(data_dir: Path, camera_id: str, track_id: int, max_gap: float, critical_gap: float) -> Optional[bytes]:
    cfg_file = data_dir / "floorplan.json"
    if not cfg_file.exists():
        return None
    try:
        import json

        from build_graph import node_id
        from floorplan.core import journey_on_plan, load_config, render_png
        from missing_camera import find_coverage_gaps, get_full_journey
        from qa_engine.engine import Store
        store = Store(data_dir)
        G = store.graph()
        if G is None or node_id(camera_id, track_id) not in G.nodes:
            return None
        journey = get_full_journey(G, camera_id, track_id)
        gaps = find_coverage_gaps(journey, max_gap)
        for g in gaps:
            g["severity"] = "critical" if g["gap_seconds"] > critical_gap else "suspicious"
        obs = []
        for f in store.obs_files():
            obs += json.loads(f.read_text(encoding="utf-8-sig"))
        cfg = load_config(cfg_file)
        return render_png(cfg, journey_on_plan(journey, gaps, cfg, obs), data_dir, title=f"{camera_id} track {track_id}")
    except Exception:                                                              # a broken floor plan must not stop the report
        return None


def build_report(data_dir, camera_id: str, track_id: int, max_gap: float = 15.0, critical_gap: float = 45.0, max_snapshots: int = 12) -> bytes:
    """Returns the PDF as bytes. Raises LookupError when there is no such sighting (same as the Investigate screen)."""
    d = Path(data_dir)
    b = web.subject_bundle(d, camera_id, track_id, max_gap, critical_gap)
    S = _styles()
    hops, gaps = b["hops"], b["gaps"]
    story: list = []
    P = lambda text, style="base": Paragraph(text, S[style])                         # noqa: E731

    # ---- title block
    story += [P("CatchCam incident report", "title"),
              P(f"Subject: {escape(camera_id)} · track {track_id} &nbsp;&nbsp;|&nbsp;&nbsp; Generated {time.strftime('%d %b %Y, %H:%M')} &nbsp;&nbsp;|&nbsp;&nbsp; Times are seconds on the shared camera clock (m:ss)", "sub"),
              Spacer(1, 8)]

    # ---- summary
    cams = [h["camera_id"] for h in hops]
    longest = max((g["gap_seconds"] for g in gaps), default=0)
    first_clothing = next((h["clothing"] for h in hops if h.get("clothing")), None)
    facts = [[P("Cameras visited", "cellb"), P(" &gt; ".join(escape(c) for c in cams), "cell")],
             [P("First / last seen", "cellb"), P(f"{_t(b['t_min'])} to {_t(b['t_max'])} ({b['t_max'] - b['t_min']:.0f}s overall)", "cell")],
             [P("Unaccounted gaps", "cellb"), P(f"{len(gaps)}" + (f", longest {longest:.0f}s" if gaps else " (none above the %.0fs normal-gap setting)" % max_gap), "cell")],
             [P("Clothing recorded", "cellb"), P(escape(_clothing_text(first_clothing)), "cell")],
             [P("People seen alongside", "cellb"), P(str(len({(c['camera_id'], c['track_id']) for c in b['companions']})) or "0", "cell")]]
    story += [P("Summary", "h"), P(escape(b["narrative"])), Spacer(1, 6), _table(facts, [4 * cm, PAGE_W - 4 * cm], header=False)]

    # ---- timeline
    story += [P("Timeline", "h"), _timeline(hops, gaps),
              P("Yellow bars are the periods the person was tracked on each camera. A dashed red line is a stretch where the system lost them "
                "between two cameras for longer than the normal-gap setting.", "small")]

    # ---- the links, with the evidence behind each
    rows = [[P(x, "cellb") for x in ("#", "Camera / track", "From", "To", "Length", "Link to previous", "Clothing check")]]
    for i, h in enumerate(hops, 1):
        link = "start of journey" if h["similarity"] is None else f"appearance match {h['similarity'] * 100:.0f}%"
        chk = (h.get("clothing_check") or {}).get("verdict")
        chk_txt = {"consistent": "consistent", "conflict": "CONFLICT: colours differ", "unknown": "not enough data"}.get(chk, "-") if h["similarity"] is not None else "-"
        rows.append([P(str(i), "cell"), P(f"{escape(h['camera_id'])} · track {h['track_id']}", "cell"), P(_t(h["start"]), "cell"), P(_t(h["end"]), "cell"),
                     P(f"{h['duration']:.0f}s", "cell"), P(link, "cell"), P(chk_txt, "cell")])
    story += [P("Journey, link by link", "h"), _table(rows, [0.8 * cm, 3.4 * cm, 1.7 * cm, 1.7 * cm, 1.5 * cm, 4.2 * cm, PAGE_W - 13.3 * cm]),
              P("Each link joins two sightings because their appearance was similar. It is evidence, not proof: a high score can still be a different person, "
                "which is why the clothing colours are compared as a second opinion.", "small")]

    # ---- snapshots
    snaps = []
    for h in hops[:max_snapshots]:
        mid = (h["start"] + h["end"]) / 2
        jpg = web.frame_jpeg(d, h["camera_id"], mid, h["track_id"], True, 560) if h.get("has_video") else None
        caption = P(f"<b>{escape(h['camera_id'])}</b> · track {h['track_id']} · {_t(mid)}", "small")
        if jpg:
            w = PAGE_W / 2 - 0.3 * cm
            img = Image(BytesIO(jpg))
            img.drawHeight, img.drawWidth = w * img.imageHeight / img.imageWidth, w
            snaps.append([img, caption])
        else:
            snaps.append([P("No stored footage for this camera.", "small"), caption])
    if snaps:
        cells = []
        for i in range(0, len(snaps), 2):
            row = [Table([[snaps[i][0]], [snaps[i][1]]], colWidths=[PAGE_W / 2 - 0.2 * cm])]
            row.append(Table([[snaps[i + 1][0]], [snaps[i + 1][1]]], colWidths=[PAGE_W / 2 - 0.2 * cm]) if i + 1 < len(snaps) else "")
            cells.append(row)
        t = Table(cells, colWidths=[PAGE_W / 2, PAGE_W / 2])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story += [P("Snapshots", "h"), t]
        if len(hops) > max_snapshots:
            story.append(P(f"Showing the first {max_snapshots} of {len(hops)} sightings.", "small"))

    # ---- gaps
    if gaps:
        rows = [[P(x, "cellb") for x in ("From", "To", "Last seen", "Next seen", "Unaccounted", "Level")]]
        for g in gaps:
            rows.append([P(f"{escape(g['from_camera'])} · track {g['from_track']}", "cell"), P(f"{escape(g['to_camera'])} · track {g['to_track']}", "cell"),
                         P(_t(g["last_seen"]), "cell"), P(_t(g["next_seen"]), "cell"), P(f"{g['gap_seconds']:.0f}s", "cell"), P(escape(g["severity"]), "cell")])
        story += [KeepTogether([P("Unaccounted time", "h"), _table(rows, [3.6 * cm, 3.6 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm, PAGE_W - 14.2 * cm]),
                                P("Unaccounted means no camera tracked the person in that stretch. It may be a blind spot, a missed detection or a wrong link; "
                                  "it does not by itself mean anything happened.", "small")])]

    # ---- companions
    if b["companions"]:
        rows = [[P(x, "cellb") for x in ("Camera", "Person (track)", "Together from", "to", "For", "Clothing")]]
        for c in b["companions"][:15]:
            rows.append([P(escape(c["camera_id"]), "cell"), P(f"track {c['track_id']}", "cell"), P(_t(c["start"]), "cell"), P(_t(c["end"]), "cell"),
                         P(f"{c['duration']:.0f}s", "cell"), P(escape(_clothing_text(c.get("clothing"))), "cell")])
        story += [KeepTogether([P("Seen alongside", "h"), _table(rows, [2.4 * cm, 2.6 * cm, 2.6 * cm, 2.0 * cm, 1.8 * cm, PAGE_W - 11.4 * cm])])]

    # ---- floor plan
    png = _floor_plan_png(d, camera_id, track_id, max_gap, critical_gap)
    if png:
        img = Image(BytesIO(png))
        img.drawWidth, img.drawHeight = PAGE_W, PAGE_W * img.imageHeight / img.imageWidth
        story += [KeepTogether([P("Floor plan", "h"), img])]

    # ---- limits
    story += [P("How to read this report", "h"),
              P("Everything here is produced automatically from camera tracking and appearance matching. Nothing has been checked by a person. "
                "Appearance matching can confuse people who dress alike, and clothing colours are approximate and only reliable within one day. "
                "CatchCam does not identify anyone by face or name. Before acting on a finding, watch the footage for the times listed above.", "small")]

    buf = BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, f"CatchCam · {camera_id} track {track_id}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title=f"CatchCam incident report: {camera_id} track {track_id}", author="CatchCam")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
