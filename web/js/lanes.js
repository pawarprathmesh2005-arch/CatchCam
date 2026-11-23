// The cross-camera timeline: one lane per camera, a yellow bar wherever the person was seen,
// and a dashed red line across any stretch where they were not on any camera.
import { svg, fmt } from './util.js';

const W = 1000, LEFT = 74, RIGHT = 14, LANE_H = 30, LANE_GAP = 8, TOP = 8, AXIS_H = 24;
const STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];

export function drawLanes(bundle, handlers) {
  const { hops, gaps } = bundle;
  const lanes = [...new Set(hops.map(x => x.camera_id))];                 // cameras in the order the person met them
  const span = Math.max(bundle.t_max - bundle.t_min, 1);
  const pad = Math.max(span * 0.06, 1.5);
  const t0 = bundle.t_min - pad, t1 = bundle.t_max + pad;
  const x = t => LEFT + ((t - t0) / (t1 - t0)) * (W - LEFT - RIGHT);
  const laneY = cam => TOP + lanes.indexOf(cam) * (LANE_H + LANE_GAP);
  const H = TOP + lanes.length * (LANE_H + LANE_GAP) + AXIS_H;
  const toTime = (evt, el) => {
    const r = el.getBoundingClientRect();
    const vx = ((evt.clientX - r.left) / r.width) * W;
    return t0 + ((vx - LEFT) / (W - LEFT - RIGHT)) * (t1 - t0);
  };

  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Timeline of the person across cameras' });

  // lane backgrounds (clicking one jumps that camera to that moment)
  lanes.forEach(cam => {
    const y = laneY(cam);
    root.append(svg('text', { class: 'lane-label', x: 0, y: y + LANE_H / 2 + 4 }, cam));
    root.append(svg('rect', { class: 'lane-bg', x: LEFT, y, width: W - LEFT - RIGHT, height: LANE_H, rx: 4 }));
    const hit = svg('rect', { class: 'hit', x: LEFT, y, width: W - LEFT - RIGHT, height: LANE_H });
    hit.addEventListener('click', e => handlers.onSeek(cam, toTime(e, root)));
    root.append(hit);
  });

  // time axis
  const step = STEPS.find(s => (t1 - t0) / s <= 8) || STEPS[STEPS.length - 1];
  const axisY = TOP + lanes.length * (LANE_H + LANE_GAP);
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    root.append(svg('line', { class: 'tick', x1: x(t), x2: x(t), y1: TOP, y2: axisY }));
    root.append(svg('text', { class: 'tick-label', x: x(t), y: axisY + 15, 'text-anchor': 'middle' }, fmt(t)));
  }

  // links between consecutive sightings: dashed red if the person was unaccounted for, thin yellow if not
  hops.slice(0, -1).forEach((a, i) => {
    const b = hops[i + 1];
    const gap = gaps.find(g => Math.abs(g.last_seen - a.end) < 0.05 && Math.abs(g.next_seen - b.start) < 0.05);
    const x1 = x(a.end), y1 = laneY(a.camera_id) + LANE_H / 2, x2 = x(b.start), y2 = laneY(b.camera_id) + LANE_H / 2;
    root.append(svg('line', { class: gap ? 'gap' : 'link', x1, y1, x2, y2 }));
    if (gap) {
      const label = svg('text', { class: 'gap-label', x: (x1 + x2) / 2 + 6, y: (y1 + y2) / 2 - 6 }, `${gap.gap_seconds.toFixed(0)}s unaccounted`);
      root.append(label);
    }
  });

  // the sightings themselves
  const segs = hops.map((hop, i) => {
    const r = svg('rect', { class: 'seg', x: x(hop.start), y: laneY(hop.camera_id) + 4, width: Math.max(x(hop.end) - x(hop.start), 5),
      height: LANE_H - 8, rx: 3, tabindex: 0, role: 'button', 'aria-label': `${hop.camera_id}, ${fmt(hop.start)} to ${fmt(hop.end)}` },
    svg('title', {}, `${hop.camera_id} · track ${hop.track_id} · ${fmt(hop.start)}–${fmt(hop.end)}`));
    r.addEventListener('click', () => handlers.onHop(i));
    r.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handlers.onHop(i); } });
    root.append(r);
    return r;
  });

  const head = svg('line', { class: 'playhead', x1: -10, x2: -10, y1: TOP, y2: axisY, visibility: 'hidden' });
  root.append(head);

  return {
    root,
    setPlayhead(t) {
      if (t == null || t < t0 || t > t1) { head.setAttribute('visibility', 'hidden'); return; }
      head.setAttribute('x1', x(t)); head.setAttribute('x2', x(t)); head.setAttribute('visibility', 'visible');
    },
    setCurrent(i) { segs.forEach((s, k) => s.style.stroke = k === i ? '#fff' : 'none'); segs.forEach(s => s.style.strokeWidth = '2'); },
  };
}
