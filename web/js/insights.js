// Insights: the numbers behind the footage, and the places that need a human to look.
import { h, clear, api, fmt, toast, svg } from './util.js';
import { mountAccuracy } from './accuracy.js';
import { mountAlerts } from './alerts.js';

let root;

export function mount(el) { root = el; }

export async function show() {
  let o;
  try { o = await api('/web/overview'); }
  catch (e) { clear(root).append(h('p', { class: 'empty' }, e.message)); toast(e.message, 'error'); return; }
  render(o);
}

function stat(value, label, alert = false) { return h('div', { class: `stat${alert ? ' alert' : ''}` }, h('b', {}, String(value)), h('span', {}, label)); }

function render(o) {
  clear(root).append(h('div', { class: 'page-head' }, h('h2', {}, 'Insights'), h('p', {}, 'What the cameras saw, and what needs a human to look.')));
  if (!o.cameras.length) {
    root.append(h('div', { class: 'panel' }, h('p', { class: 'empty' }, 'Nothing to show until there is footage.'), h('a', { class: 'btn primary', href: '#cameras' }, 'Add footage')));
    return;
  }
  const down = o.cameras_down.length;
  root.append(h('div', { class: 'stats' }, stat(o.cameras.length, 'cameras'), stat(down, down === 1 ? 'camera down' : 'cameras down', down > 0),
    stat(o.people_tracked, 'people tracked'), stat(o.matches, 'cross-camera matches'), stat(o.with_clothing_data, 'with clothing data'), stat(o.vehicles, 'vehicles')));

  const alertsPanel = h('section', { class: 'panel', id: 'alerts-panel' });
  root.append(alertsPanel);
  mountAlerts(alertsPanel);

  const max = Math.max(1, ...Object.values(o.activity).flat());
  const bars = h('div', { class: 'bars' });
  Object.entries(o.activity).forEach(([cam, vals]) => {
    const w = 240, bw = w / vals.length;
    const s = svg('svg', { viewBox: `0 0 ${w} 40`, preserveAspectRatio: 'none', role: 'img', 'aria-label': `${cam} activity over time` },
      ...vals.map((v, i) => svg('rect', { x: i * bw + 0.5, y: 40 - (v / max) * 38, width: Math.max(bw - 1, 1), height: (v / max) * 38, fill: 'var(--link)', opacity: 0.85, rx: 1 },
        svg('title', {}, `${v} ${v === 1 ? 'person' : 'people'} appeared`))));
    bars.append(h('div', { class: 'r' }, h('span', {}, cam), s));
  });
  const when = o.t_min != null ? h('p', { class: 'muted', style: 'font-size:13px;margin-top:8px' }, `From ${fmt(o.t_min)} to ${fmt(o.t_max)} on the shared clock.`) : null;

  const flagged = o.flagged_journeys.length
    ? h('ul', { class: 'list' }, o.flagged_journeys.map(f => h('li', {}, h('div', { class: 'grow' }, h('div', { class: 'who' }, f.route), h('div', { class: 'sub' },
      `${f.gaps} gap${f.gaps > 1 ? 's' : ''}, longest ${f.longest_gap.toFixed(0)}s · starts at ${f.camera_id} track ${f.track_id}`)),
    h('a', { class: 'btn small', href: `#investigate/${encodeURIComponent(f.camera_id)}/${f.track_id}` }, 'Investigate'))))
    : h('p', { class: 'empty' }, 'No journey has an unexplained gap.');
  const conflicts = o.match_conflicts.length
    ? h('ul', { class: 'list' }, o.match_conflicts.map(m => h('li', {}, h('div', { class: 'grow' }, h('div', { class: 'who' }, `${m.a} ↔ ${m.b}`),
      h('div', { class: 'sub' }, `Appearance match ${(m.similarity * 100).toFixed(0)}%, but the clothing colours disagree. This may be a wrong match.`)))))
    : h('p', { class: 'empty' }, 'No matches where the clothing disagrees.');

  root.append(h('div', { class: 'two' },
    h('section', { class: 'panel' }, h('h3', {}, 'When people appeared'), bars, when),
    h('section', { class: 'panel' }, h('h3', {}, 'Journeys with unexplained gaps'), flagged)),
  h('section', { class: 'panel' }, h('h3', {}, 'Matches to double-check'), conflicts));
  const acc = h('section', { class: 'panel', id: 'accuracy' });
  root.append(acc);
  mountAccuracy(acc);
}
