// The Investigate screen: the camera wall, ONE search box, and everything about whoever you pick.
// Flow: type or say anything -> /web/find -> results (and, if it is clear who you mean, we open them straight away)
//       -> /web/subject -> video follows the person camera to camera, timeline + tabs describe the journey.
import { $, h, clear, api, fmt, clothingLine, toast, speak, listen } from './util.js';
import { makeSurface } from './player.js';
import { drawLanes } from './lanes.js';

const FALLBACK_EXAMPLES = ['Where did the person in the red jacket go?', 'Show the journey of the person in a blue shirt and black pants',
  'Where did track 5 on cam1 come from?', 'Which cameras are down?'];

const S = {
  cams: [], ctx: null, results: [], bundle: null, hopIdx: 0, follow: false, paused: false, rate: 1, tab: 'journey',
  surface: null, wall: [], layout: 0, hidden: new Set(), page: 0, lanes: null, timer: null, bannerTimer: null, examples: FALLBACK_EXAMPLES, selectedKey: null,
};
let el = {};
const MAX_LIVE = 4;                  // cameras playing at once on the wall; the rest are pictures with a '▶ Live' button

// remembered between visits (never required: private windows can block storage)
const PREF = 'catchcam.wall';
try {
  const p = JSON.parse(localStorage.getItem(PREF) || '{}');
  if ([0, 1, 2, 4, 6].includes(p.layout)) S.layout = p.layout;
  if (Array.isArray(p.hidden)) S.hidden = new Set(p.hidden);
} catch { /* ignore */ }
const savePref = () => { try { localStorage.setItem(PREF, JSON.stringify({ layout: S.layout, hidden: [...S.hidden] })); } catch { /* ignore */ } };

// ----------------------------------------------------------------------------- mount / show / hide
export function mount(root) {
  const input = h('input', { class: 'input', id: 'q', type: 'text', enterkeyhint: 'search', autocomplete: 'off', placeholder: 'Describe a person or car, give a plate or track, or ask a question…',
    'aria-label': 'Search' });
  const mic = h('button', { class: 'mic', type: 'button', 'aria-label': 'Speak your question', title: 'Speak your question' }, micIcon());
  mic.addEventListener('click', () => {
    if (mic._rec) { mic._rec.stop(); return; }
    mic._rec = listen(text => { input.value = text; runSearch(text); }, on => { mic.classList.toggle('on', on); if (!on) mic._rec = null; });
  });
  const form = h('form', { class: 'search', role: 'search' }, h('div', { class: 'box' }, input, mic), h('button', { class: 'btn primary', type: 'submit' }, 'Search'));
  form.addEventListener('submit', e => { e.preventDefault(); runSearch(input.value); });

  el = { input, mic, answer: h('div', { id: 'answer' }), stage: h('div', { class: 'stage', id: 'stage' }), lanes: h('div', { id: 'lanes' }),
    describe: h('div', { id: 'describe' }), rail: h('aside', { class: 'rail', id: 'rail', 'aria-label': 'Results' }) };
  root.append(form, el.answer, h('div', { class: 'investigate' }, h('div', { class: 'col-main' }, el.stage, el.lanes, el.describe), el.rail));
  api('/qa/examples').then(r => { if (r.examples?.length) { S.examples = r.examples; if (!S.results.length) renderRail(); } }).catch(() => {});
  window.addEventListener('catchcam:cameras-changed', () => { if (!root.hidden) refreshCameras().then(() => { if (!S.bundle && !S.focusCam) showWall(); }); });
}

export async function show(a, b) {
  await refreshCameras();
  if (a && b !== undefined && !Number.isNaN(Number(b))) return openSubject(a, Number(b));
  if (a) return focusCamera(a);
  if (S.bundle) return presentBundle(S.bundle);                    // coming back to the tab: players were stopped, so restart them
  if (S.focusCam) return focusCamera(S.focusCam);
  showWall();
  renderRail();
}

export function hide() { stopAll(); }

async function refreshCameras() {
  try { S.cams = (await api('/web/cameras')).cameras; } catch (e) { S.cams = []; toast(e.message, 'error'); }
}
const camById = id => S.cams.find(c => c.camera_id === id) || { camera_id: id, offset: 0 };

// ----------------------------------------------------------------------------- the wall (nothing selected yet)
function stopAll() {
  clearInterval(S.timer); S.timer = null; clearTimeout(S.bannerTimer); S.banner = false;
  S.surface?.destroy(); S.surface = null;
  S.wall.forEach(s => s.destroy()); S.wall = [];
}

function showWall() {
  stopAll();
  S.bundle = null; S.focusCam = null; S.follow = false;
  clear(el.lanes);
  if (!S.cams.length) {
    clear(el.stage).append(h('div', { class: 'panel' }, h('h2', {}, 'No cameras yet'),
      h('p', { class: 'muted', style: 'margin:6px 0 12px' }, 'Add a recorded video or a live camera, and CatchCam will start finding and following people.'),
      h('a', { class: 'btn primary', href: '#cameras' }, 'Add footage')));
    clear(el.describe);
    return;
  }
  const pool = S.cams.filter(c => !S.hidden.has(c.camera_id));
  const size = S.layout || Math.max(pool.length, 1);
  const pages = Math.max(1, Math.ceil(pool.length / size));
  S.page = Math.min(S.page, pages - 1);
  const shown = pool.slice(S.page * size, (S.page + 1) * size);
  const wall = h('div', { class: `wall n${Math.min(Math.max(shown.length, 1), 9)}`, id: 'wall' });
  if (!shown.length) wall.append(h('p', { class: 'empty' }, 'No cameras selected. Pick one or more above.'));
  // Only a few cameras play at once (see player.js: browsers allow ~6 connections per server); the rest are pictures.
  const live = [];
  shown.forEach((c, i) => {
    const startsLive = i < MAX_LIVE;
    const surf = makeSurface(c, { loop: true, width: 560, still: !startsLive });
    S.wall.push(surf);
    const entry = { surf, go: h('span', { class: 'golive', role: 'button', tabindex: '0', hidden: startsLive,
      title: `Watch ${c.camera_id} live (another camera goes back to a picture)` }, '▶ Live') };
    const promote = e => {
      e.stopPropagation(); e.preventDefault();
      if (live.length >= MAX_LIVE) { const old = live.shift(); old.surf.goStill(); old.go.hidden = false; }
      surf.goLive(); live.push(entry); entry.go.hidden = true;
    };
    entry.go.addEventListener('click', promote);
    entry.go.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') promote(e); });
    const tile = h('button', { class: 'tile', type: 'button', 'aria-label': `Open ${c.camera_id}` }, surf.root,
      h('span', { class: 'tag' }, h('i', { class: `led ${c.status}` }), c.camera_id),
      h('span', { class: 'count' }, c.num_tracks ? `${c.num_tracks} ${c.num_tracks === 1 ? 'person' : 'people'}` : 'no people found yet'),
      shown.length > MAX_LIVE ? entry.go : null);
    tile.addEventListener('click', () => { location.hash = `#investigate/${encodeURIComponent(c.camera_id)}`; });
    wall.append(tile);
    if (startsLive) { live.push(entry); surf.play(); }
  });
  clear(el.stage).append(wallControls(pool.length, pages), wall,
    shown.length > MAX_LIVE ? h('p', { class: 'muted wall-note' }, `${MAX_LIVE} cameras play live; the others are pictures, because a browser can only stream a few videos at once. Press ▶ Live on a picture to watch it, or click a camera to look closer.`) : null);
  clear(el.describe).append(h('div', { class: 'describe' }, h('div', { class: 'describe-head' },
    h('p', {}, S.cams.some(c => c.num_tracks) ? 'All your cameras are shown above. Search for someone, or click a camera to look closer.'
      : 'Your cameras are shown above. No people have been found yet: open Cameras and press Process on a camera.'))));
}

function wallControls(poolSize, pages) {
  const redraw = () => { savePref(); showWall(); };
  const sizes = h('div', { class: 'seg-ctl', role: 'group', 'aria-label': 'Cameras shown at once' }, h('span', { class: 'muted' }, 'Show'),
    [[1, '1'], [2, '2'], [4, '4'], [6, '6'], [0, 'All']].map(([n, label]) => h('button', { class: 'btn small ghost', type: 'button', 'data-layout': n,
      'aria-pressed': String(S.layout === n), onClick: () => { S.layout = n; S.page = 0; redraw(); } }, label)));
  const chips = h('div', { class: 'cam-chips', role: 'group', 'aria-label': 'Choose cameras' },
    S.cams.map(c => h('button', { class: 'chip cam', type: 'button', 'data-cam': c.camera_id, 'aria-pressed': String(!S.hidden.has(c.camera_id)),
      title: 'Click to show or hide on the wall; double-click to watch only this one',
      onClick: () => { if (S.hidden.has(c.camera_id)) S.hidden.delete(c.camera_id); else S.hidden.add(c.camera_id); S.page = 0; redraw(); },
      onDblclick: () => { location.hash = `#investigate/${encodeURIComponent(c.camera_id)}`; } }, h('i', { class: `led ${c.status}` }), c.camera_id)),
    h('button', { class: 'chip', type: 'button', id: 'cam-all', onClick: () => { S.hidden.clear(); S.page = 0; redraw(); } }, 'Select all'));
  const bar = h('div', { class: 'wall-bar' }, sizes, chips);
  if (pages > 1) {
    bar.append(h('div', { class: 'pager' },
      h('button', { class: 'btn small', type: 'button', id: 'pg-prev', 'aria-label': 'Previous cameras', onClick: () => { S.page = (S.page - 1 + pages) % pages; showWall(); } }, '‹'),
      h('span', { class: 'muted', id: 'pg-n' }, `${S.page + 1} / ${pages}`),
      h('button', { class: 'btn small', type: 'button', id: 'pg-next', 'aria-label': 'Next cameras', onClick: () => { S.page = (S.page + 1) % pages; showWall(); } }, '›')));
  }
  return bar;
}

// one row of buttons to jump to any camera while looking closely at one (no need to go back to the wall)
function cameraSwitcher(currentId) {
  return h('div', { class: 'cam-chips switch', role: 'group', 'aria-label': 'Switch camera' },
    S.cams.map(c => h('button', { class: 'chip cam', type: 'button', 'data-cam': c.camera_id, 'aria-pressed': String(c.camera_id === currentId),
      onClick: () => { if (c.camera_id !== currentId) location.hash = `#investigate/${encodeURIComponent(c.camera_id)}`; } }, h('i', { class: `led ${c.status}` }), c.camera_id)));
}

// ----------------------------------------------------------------------------- one camera, nobody selected
function focusCamera(id, at = null) {
  stopAll();
  S.bundle = null; S.follow = false; S.focusCam = id;
  clear(el.lanes); clear(el.describe);
  const cam = camById(id);
  const strip = S.cams.map(c => ({ cam: c, current: c.camera_id === id, onClick: () => { location.hash = `#investigate/${encodeURIComponent(c.camera_id)}`; } }));
  buildFocus({ cam, startAt: at, track: null, strip, loop: true, switcher: cameraSwitcher(id) });
  el.describe.append(h('div', { class: 'describe' }, h('div', { class: 'describe-head' },
    h('p', {}, `${cam.camera_id}: ${cam.num_tracks ? `${cam.num_tracks} people found` : 'no people found yet'}. Boxes mark everyone the system tracks. Search above to follow one person.`))));
}

// ----------------------------------------------------------------------------- the video player (shared by both modes)
function buildFocus({ cam, startAt, track, strip, loop, subjectMode = false, switcher = null }) {
  const playerBox = h('div', { class: 'player' });
  const stripBox = h('div', { class: 'strip' }, strip.map(s => stripTile(s)));
  const now = h('span', { class: 'now', id: 'now' });
  const playBtn = h('button', { class: 'btn small', type: 'button', id: 'play' }, 'Pause');
  const rates = [1, 2, 4].map(r => {
    const b = h('button', { class: 'btn small ghost', type: 'button', 'aria-pressed': String(r === S.rate) }, `${r}×`);
    b.addEventListener('click', () => { S.rate = r; S.surface?.setRate(r); rates.forEach(x => x.setAttribute('aria-pressed', String(x === b))); });
    return b;
  });
  const bar = h('div', { class: 'player-bar' });
  if (subjectMode) {
    bar.append(h('button', { class: 'btn small ghost', type: 'button', onClick: () => { location.hash = '#investigate'; S.bundle = null; showWall(); renderRail(); } }, '← All cameras'));
  } else {
    bar.append(h('a', { class: 'btn small ghost', href: '#investigate', onClick: () => { S.bundle = null; S.focusCam = null; } }, '← All cameras'));
  }
  bar.append(playBtn, h('span', { class: 'muted', style: 'font-size:13px' }, 'Speed'), ...rates);
  if (subjectMode) {
    const follow = h('button', { class: 'btn small', type: 'button', id: 'follow', 'aria-pressed': String(S.follow) }, S.follow ? 'Following journey ✓' : 'Follow journey');
    follow.addEventListener('click', () => {
      S.follow = !S.follow; follow.textContent = S.follow ? 'Following journey ✓' : 'Follow journey'; follow.setAttribute('aria-pressed', String(S.follow));
      if (S.follow) { S.paused = false; S.surface?.play(); playBtn.textContent = 'Pause'; }
    });
    bar.append(follow);
  }
  bar.append(now);
  playBtn.addEventListener('click', () => {
    S.paused = !S.paused;
    if (S.paused) S.surface?.pause(); else S.surface?.play();
    playBtn.textContent = S.paused ? 'Play' : 'Pause';
  });
  clear(el.stage).append(...(switcher ? [switcher] : []), h('div', { class: 'focus' }, playerBox, stripBox), bar);
  el.playerBox = playerBox; el.now = now; el.playBtn = playBtn; el.stripBox = stripBox;
  S.paused = false;
  setPlayer(cam, startAt, track, loop);
  clearInterval(S.timer);
  S.timer = setInterval(tick, 150);
}

function stripTile({ cam, current, onClick, label }) {
  const t = cam.first_seen != null ? cam.first_seen + 1 : 1;
  const img = h('img', { src: `/web/frame?camera_id=${encodeURIComponent(cam.camera_id)}&t=${t}&width=320`, alt: '', loading: 'lazy' });
  img.addEventListener('error', () => img.replaceWith(h('div', { class: 'nofeed', style: 'font-size:12px' }, 'no picture')));
  const b = h('button', { class: `tile${current ? ' current' : ''}`, type: 'button', 'aria-label': `Show ${cam.camera_id}` },
    h('div', { class: 'surface' }, img), h('span', { class: 'tag' }, label || cam.camera_id));
  b.addEventListener('click', onClick);
  return b;
}

function setPlayer(cam, at, track, loop = false) {
  S.surface?.destroy();
  const surf = makeSurface(cam, { controls: true, loop, width: 960, startAt: at, track });
  S.surface = surf;
  clear(el.playerBox).append(surf.root);
  surf.setRate(S.rate);
  surf.play();
  if (S.paused) surf.pause();
}

function tick() {
  if (!S.surface || !el.now?.isConnected) return;
  const t = S.surface.currentShared();
  const b = S.bundle;
  el.now.textContent = `${S.surface.cam.camera_id} · ${fmt(t, true)}`;
  if (!b) return;
  S.lanes?.setPlayhead(t);
  if (!S.follow || S.paused || S.banner) return;
  const hop = b.hops[S.hopIdx];
  if (t >= hop.end - 0.03 || S.surface.ended) advance();
}

function advance() {
  const b = S.bundle, i = S.hopIdx, hop = b.hops[i], next = b.hops[i + 1];
  if (!next) { S.follow = false; S.surface?.pause(); refreshFollowButton(); toast('End of this person\'s journey.'); return; }
  const gap = b.gaps.find(g => Math.abs(g.last_seen - hop.end) < 0.05 && Math.abs(g.next_seen - next.start) < 0.05);
  if (!gap) { goHop(i + 1); return; }
  S.surface?.pause();
  S.banner = true;
  const box = h('div', { class: 'banner', role: 'status' }, `Not seen for ${gap.gap_seconds.toFixed(0)}s between ${gap.from_camera} and ${gap.to_camera}`);
  el.playerBox.append(box);
  S.bannerTimer = setTimeout(() => { box.remove(); S.banner = false; goHop(i + 1); }, 2200);
}

function refreshFollowButton() {
  const f = $('#follow');
  if (f) { f.textContent = S.follow ? 'Following journey ✓' : 'Follow journey'; f.setAttribute('aria-pressed', String(S.follow)); }
}

function goHop(i, at = null) {
  const b = S.bundle, hop = b.hops[i];
  clearTimeout(S.bannerTimer); S.banner = false;                    // a pending "not seen for…" banner must not fire after a manual jump
  S.hopIdx = i;
  const cam = camById(hop.camera_id);
  const start = at ?? Math.max(hop.start - 0.5, cam.offset || 0);
  setPlayer(cam, start, b.kind === 'vehicle' ? null : hop.track_id, false);
  S.paused = false; if (el.playBtn) el.playBtn.textContent = 'Pause';
  $('.player', el.stage)?.append(h('div', { class: 'hop-badge' }, `Sighting ${i + 1} of ${b.hops.length} · ${hop.camera_id}`));
  el.stripBox?.querySelectorAll('.tile').forEach((t, k) => t.classList.toggle('current', S.stripCams[k] === hop.camera_id));
  S.lanes?.setCurrent(i);
  if (S.tab === 'journey') renderTab();
}

// ----------------------------------------------------------------------------- a person (or vehicle) is selected
async function openSubject(cam, track) {
  let b;
  try { b = await api('/web/subject', { camera_id: cam, track_id: track }); }
  catch (e) { toast(e.message, 'error'); showWall(); renderRail(); return; }
  b.kind = 'person';
  S.selectedKey = `${cam}:${track}`;
  S.ctx = { camera_id: cam, track_id: track };
  presentBundle(b);
}

async function openVehicle(card) {
  const rows = S.results.filter(r => r.kind === 'vehicle' && (card.plate ? r.plate === card.plate : r === card));
  let j = null;
  if (card.plate) { try { j = await api('/vehicles/journey', { plate: card.plate }); } catch { /* fall back to the single sighting */ } }
  const sightings = (j?.sightings?.length ? j.sightings : rows.length ? rows.map(r => ({ camera_id: r.camera_id, track_id: r.track_id, first_seen_timestamp: r.first_seen, last_seen_timestamp: r.last_seen })) : [])
    .filter(r => r.first_seen_timestamp != null);
  if (!sightings.length) { toast('No footage position is stored for that vehicle.', 'error'); return; }
  const hops = sightings.map(r => ({ camera_id: r.camera_id, track_id: r.track_id, start: r.first_seen_timestamp, end: r.last_seen_timestamp ?? r.first_seen_timestamp,
    duration: (r.last_seen_timestamp ?? r.first_seen_timestamp) - r.first_seen_timestamp, similarity: null, clothing_check: null, clothing: null,
    is_selected: false, has_video: true, thumb_url: `/web/thumb?camera_id=${r.camera_id}&track_id=${r.track_id}` }));
  const gaps = (j?.gaps || []).map(g => ({ ...g, severity: g.gap_seconds > 45 ? 'critical' : 'suspicious' }));
  S.selectedKey = `plate:${card.plate || card.track_id}`;
  S.ctx = card.plate ? { plate: card.plate } : S.ctx;
  presentBundle({ kind: 'vehicle', camera_id: hops[0].camera_id, track_id: hops[0].track_id, hops, gaps, has_blind_spot: gaps.length, companions: [],
    narrative: S.lastAnswer || `${card.plate || 'This vehicle'} was seen on ${[...new Set(hops.map(x => x.camera_id))].join(', ')}.`,
    speak: S.lastSpeak || '', cameras: [...new Set(hops.map(x => x.camera_id))], t_min: Math.min(...hops.map(x => x.start)),
    t_max: Math.max(...hops.map(x => x.end)), floorplan_url: null });
}

function presentBundle(b) {
  stopAll();
  S.bundle = b; S.focusCam = null; S.follow = true; S.hopIdx = 0; S.tab = 'journey';
  const hops = b.hops;
  const strip = [...new Set(hops.map(x => x.camera_id))].map(id => ({ cam: camById(id), current: false, onClick: () => { const i = hops.findIndex(x => x.camera_id === id); goHop(i); } }));
  S.stripCams = strip.map(s => s.cam.camera_id);
  const first = hops[0], cam = camById(first.camera_id);
  buildFocus({ cam, startAt: Math.max(first.start - 0.5, cam.offset || 0), track: b.kind === 'vehicle' ? null : first.track_id, strip, loop: false, subjectMode: true });
  $('.player', el.stage).append(h('div', { class: 'hop-badge' }, `Sighting 1 of ${hops.length} · ${first.camera_id}`));
  el.stripBox.querySelector('.tile')?.classList.add('current');
  S.lanes = drawLanes(b, { onHop: i => goHop(i), onSeek: (c, t) => seekOn(c, t) });
  S.lanes.setCurrent(0);
  clear(el.lanes).append(h('div', { class: 'lanes' }, S.lanes.root,
    h('div', { class: 'lanes-cap' }, h('span', { class: 'key' }, h('i'), 'seen on camera'), h('span', { class: 'key' }, h('i', { class: 'gap' }), 'not on any camera'),
      h('span', {}, 'Click a bar to jump to that sighting, or a lane to jump to a moment.'))));
  renderDescribe();
  renderRail();
  el.stage.scrollIntoView?.({ block: 'nearest' });
}

function seekOn(camId, t) {
  const b = S.bundle;
  const i = b.hops.findIndex(x => x.camera_id === camId && t >= x.start - 0.2 && t <= x.end + 0.2);
  if (i >= 0) { S.hopIdx = i; goHop(i, t); S.lanes?.setCurrent(i); return; }
  S.follow = false; refreshFollowButton();
  setPlayer(camById(camId), t, null, false);
}

// ----------------------------------------------------------------------------- the description under the video
function renderDescribe() {
  const b = S.bundle;
  const tabs = [['journey', 'Journey', b.hops.length], ['plan', 'Floor plan'], ['with', 'Who was with them', b.companions.length], ['others', 'Same person'],
    ['clothing', 'Clothing'], ['gaps', 'Gaps', b.gaps.length, b.gaps.length > 0]];
  const visible = tabs.filter(([k]) => !(b.kind === 'vehicle' && ['plan', 'with', 'clothing', 'others'].includes(k)));
  const bar = h('div', { class: 'dtabs', role: 'tablist' }, visible.map(([k, label, n, bad]) => {
    const btn = h('button', { role: 'tab', type: 'button', 'aria-selected': String(S.tab === k), 'data-tab': k }, label, n ? h('span', { class: `n${bad ? ' bad' : ''}` }, n) : null);
    btn.addEventListener('click', () => { S.tab = k; bar.querySelectorAll('button').forEach(x => x.setAttribute('aria-selected', String(x === btn))); renderTab(); });
    return btn;
  }));
  const say = h('button', { class: 'btn small ghost', type: 'button' }, '🔊 Read aloud');
  say.addEventListener('click', () => speak(b.speak || b.narrative));
  const report = b.kind === 'vehicle' ? null : h('button', { class: 'btn small', type: 'button', id: 'export-report', title: 'A PDF with the journey, timeline, snapshots and gaps' }, 'Export report (PDF)');
  report?.addEventListener('click', () => exportReport(b, report));
  clear(el.describe).append(h('div', { class: 'describe' }, h('div', { class: 'describe-head' }, h('p', { id: 'narr' }, b.narrative), h('div', { class: 'head-actions' }, say, report)), bar, h('div', { class: 'dbody', id: 'dbody', role: 'tabpanel' })));
  renderTab();
}

// the PDF is fetched first (not a plain link) so a server error becomes a message instead of a downloaded error file
async function exportReport(b, btn) {
  btn.disabled = true; const label = btn.textContent; btn.textContent = 'Preparing…';
  try {
    const q = new URLSearchParams({ camera_id: b.camera_id, track_id: b.track_id });
    let res;
    try { res = await fetch(`/report/journey.pdf?${q}`); } catch { throw new Error("Can't reach the CatchCam server."); }
    if (!res.ok) { let d = ''; try { d = (await res.json()).detail; } catch { /* not JSON */ } throw new Error(d || `The server said ${res.status}.`); }
    const url = URL.createObjectURL(await res.blob());
    const a = h('a', { href: url, download: `catchcam_report_${b.camera_id}_track${b.track_id}.pdf` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  } catch (e) { toast(e.message, 'error'); }
  finally { btn.disabled = false; btn.textContent = label; }
}

function thumb(url) {
  const img = h('img', { class: 'thumb', src: url, alt: '', loading: 'lazy' });
  img.addEventListener('error', () => img.replaceWith(h('div', { class: 'thumb blank' }, 'no photo')));
  return img;
}

function renderTab() {
  const b = S.bundle, body = $('#dbody');
  if (!b || !body) return;
  clear(body);
  if (S.tab === 'journey') {
    const list = h('ol', { class: 'steps' });
    b.hops.forEach((hop, i) => {
      const cc = hop.clothing_check;
      list.append(h('li', { class: `step${i === S.hopIdx ? ' selected' : ''}` }, thumb(hop.thumb_url),
        h('div', {}, h('div', { class: 'who' }, `${i + 1}. ${hop.camera_id}`, b.kind === 'vehicle' ? '' : ` · track ${hop.track_id}`),
          h('div', { class: 'sub' }, `${fmt(hop.start)} → ${fmt(hop.end)} (${hop.duration.toFixed(0)}s on camera)`,
            hop.similarity != null ? ` · appearance match ${(hop.similarity * 100).toFixed(0)}%` : ''),
          clothingLine(hop.clothing),
          cc && cc.verdict !== 'unknown' ? h('span', { class: `badge ${cc.verdict === 'consistent' ? 'ok' : 'bad'}`, title: cc.reason },
            cc.verdict === 'consistent' ? 'clothing agrees with previous camera' : 'clothing differs from previous camera: check this match') : null),
        h('button', { class: 'btn small', type: 'button', onClick: () => goHop(i) }, 'Watch')));
      const gap = b.gaps.find(g => Math.abs(g.last_seen - hop.end) < 0.05 && b.hops[i + 1] && Math.abs(g.next_seen - b.hops[i + 1].start) < 0.05);
      if (gap) list.append(h('li', { class: `gaprow ${gap.severity === 'critical' ? '' : 'suspicious'}` },
        h('b', {}, gap.severity === 'critical' ? 'Blind spot' : 'Unaccounted time'), `${gap.gap_seconds.toFixed(0)}s between ${gap.from_camera} and ${gap.to_camera}`));
    });
    body.append(list);
  } else if (S.tab === 'plan') {
    if (b.floorplan_url) body.append(h('img', { class: 'plan', src: b.floorplan_url, alt: 'Floor plan with the person\'s path' }));
    else body.append(h('p', { class: 'empty' }, 'No floor plan is set up yet. Copy floorplan/floorplan.example.json to floorplan.json in your project folder and fill in your camera calibration points.'));
  } else if (S.tab === 'with') {
    if (!b.companions.length) body.append(h('p', { class: 'empty' }, 'Nobody was recorded walking together with this person.'));
    else {
      const list = h('ol', { class: 'steps' });
      b.companions.forEach(c => list.append(h('li', { class: 'step' }, thumb(c.thumb_url),
        h('div', {}, h('div', { class: 'who' }, `${c.camera_id} · track ${c.track_id}`),
          h('div', { class: 'sub' }, `Together for ${c.duration.toFixed(0)}s, ${fmt(c.start)} → ${fmt(c.end)}`), clothingLine(c.clothing)),
        h('a', { class: 'btn small', href: `#investigate/${encodeURIComponent(c.camera_id)}/${c.track_id}` }, 'Follow them'))));
      body.append(list);
    }
  } else if (S.tab === 'clothing') {
    const rows = b.hops.filter(x => x.clothing && (x.clothing.upper || x.clothing.lower));
    if (!rows.length) body.append(h('p', { class: 'empty' }, 'No clothing data yet. Process the footage (Cameras → Process) to read clothing colours.'));
    else {
      const list = h('ol', { class: 'steps' });
      rows.forEach(x => list.append(h('li', { class: 'step' }, thumb(x.thumb_url), h('div', {}, h('div', { class: 'who' }, `${x.camera_id} · track ${x.track_id}`), clothingLine(x.clothing),
        h('div', { class: 'sub' }, 'Colours are approximate and only reliable within the same day.')), h('span'))));
      body.append(list);
    }
  } else if (S.tab === 'others') {
    renderOthers(body, b);
  } else if (S.tab === 'gaps') {
    if (!b.gaps.length) body.append(h('p', { class: 'empty' }, `No unexplained gaps. Every move between cameras happened within ${b.max_normal_gap ?? 15} seconds.`));
    else b.gaps.forEach(g => body.append(h('div', { class: `gaprow ${g.severity === 'critical' ? '' : 'suspicious'}`, style: 'margin-left:0;margin-bottom:8px' },
      h('b', {}, g.severity === 'critical' ? 'Blind spot' : 'Unaccounted time'),
      `${g.gap_seconds.toFixed(1)}s between ${g.from_camera} (last seen ${fmt(g.last_seen)}) and ${g.to_camera} (next seen ${fmt(g.next_seen)})`)));
  }
}

// best guesses for the same person on every other camera; a person confirms by watching
async function renderOthers(body, b) {
  body.append(h('p', { class: 'empty' }, 'Looking…'));
  let r;
  try { r = await api('/web/follow', { camera_id: b.camera_id, track_id: b.track_id }); }
  catch (e) { clear(body).append(h('p', { class: 'empty' }, e.message || 'Could not search other cameras.')); return; }
  if (S.tab !== 'others') return;
  clear(body);
  try {
    const who = await api('/web/identity', { camera_id: b.camera_id, track_id: b.track_id });
    if (S.tab !== 'others') return;
    body.append(h('div', { class: 'who' }, `${who.person_id}: ${who.appearances} appearance${who.appearances === 1 ? '' : 's'} on ${who.cameras.join(', ')}`),
      h('p', { class: 'sub' }, 'Every track CatchCam believes is this same person, in time order. It is a best guess, so check each one.'));
    const all = h('ol', { class: 'steps' });
    who.members.forEach(x => all.append(h('li', { class: 'step' }, thumb(`/web/thumb?camera_id=${x.camera_id}&track_id=${x.track_id}`),
      h('div', {}, h('div', { class: 'who' }, `${x.camera_id} · track ${x.track_id}`), h('div', { class: 'sub' }, `${fmt(x.start)} → ${fmt(x.end)}`)),
      h('a', { class: 'btn small', href: `#investigate/${encodeURIComponent(x.camera_id)}/${x.track_id}` }, 'Watch'))));
    body.append(all);
  } catch (e) { /* no identity yet: the candidate list below still works */ }
  body.append(h('p', { class: 'sub' }, 'Best guesses for the same person on each camera, strongest first. Click Watch to check each one.'));
  if (r.missing_start_times.length) body.append(h('p', { class: 'sub' }, `Set a start time for ${r.missing_start_times.join(', ')} in Cameras so times line up.`));
  r.cameras.forEach(c => {
    body.append(h('div', { class: 'who' }, c.camera_id));
    const list = h('ol', { class: 'steps' });
    c.candidates.forEach(x => list.append(h('li', { class: 'step' }, thumb(x.thumb_url),
      h('div', {}, h('div', { class: 'who' }, `track ${x.track_id} · ${Math.round(x.score * 100)}% likely`),
        h('div', { class: 'sub' }, `${fmt(x.start)} → ${fmt(x.end)}`), clothingLine(x.clothing)),
      h('a', { class: 'btn small', href: `#investigate/${encodeURIComponent(x.camera_id)}/${x.track_id}` }, 'Watch'))));
    body.append(list);
  });
}

// ----------------------------------------------------------------------------- search
async function runSearch(text) {
  const q = (text || '').trim();
  if (!q) return;
  const p = { q };
  if (S.ctx?.plate) p.context_plate = S.ctx.plate;
  else if (S.ctx?.camera_id != null) { p.context_camera = S.ctx.camera_id; p.context_track = S.ctx.track_id; }
  let r;
  try { r = await api('/web/find', p); } catch (e) { toast(e.message, 'error'); return; }
  S.results = r.results || [];
  S.lastAnswer = r.answer; S.lastSpeak = r.speak;
  if (r.context) S.ctx = r.context;
  renderAnswer(r, q);
  renderRail();
  const sel = r.selected;
  if (sel?.kind === 'person' && sel.camera_id != null) await openSubject(sel.camera_id, sel.track_id);
  else if (sel?.kind === 'vehicle') { const card = S.results.find(x => x.kind === 'vehicle' && x.plate === sel.plate) || S.results[0]; if (card) await openVehicle({ ...card, plate: sel.plate }); }
  else if (!S.bundle && !S.focusCam && S.results.length === 0) { /* answer text already explains */ }
}

function renderAnswer(r, q) {
  clear(el.answer);
  const text = r.answer || (r.results?.length ? `${r.results.length} possible ${r.results.length === 1 ? 'match' : 'matches'} for “${q}”. Pick one on the right.` : `Nothing matched “${q}”.`);
  const warn = /unaccounted|blind spot|warning|down:/i.test(text);
  const say = h('button', { class: 'btn small ghost', type: 'button' }, '🔊');
  say.setAttribute('aria-label', 'Read the answer aloud');
  say.addEventListener('click', () => speak(r.speak || text));
  el.answer.append(h('div', { class: `answer${warn ? ' warn' : ''}`, role: 'status' }, h('p', {}, text), say));
}

function renderRail() {
  clear(el.rail);
  if (!S.results.length) {
    el.rail.append(h('h3', {}, 'Try asking'), h('div', { class: 'chips', style: 'margin-top:0' }, S.examples.map(ex => {
      const c = h('button', { class: 'chip', type: 'button' }, ex);
      c.addEventListener('click', () => { el.input.value = ex; runSearch(ex); });
      return c;
    })));
    return;
  }
  el.rail.append(h('h3', {}, `${S.results.length} ${S.results.length === 1 ? 'result' : 'results'}`));
  S.results.forEach(r => {
    const key = r.kind === 'vehicle' ? `plate:${r.plate || r.track_id}` : `${r.camera_id}:${r.track_id}`;
    const inChain = S.bundle && r.kind === 'person' && S.bundle.hops.some(x => x.camera_id === r.camera_id && x.track_id === r.track_id);
    const card = h('button', { class: `card${S.selectedKey === key || inChain ? ' selected' : ''}`, type: 'button' }, thumb(r.thumb_url),
      h('div', {}, h('div', { class: 'who' }, r.kind === 'vehicle' ? (r.title || 'vehicle') : r.title),
        r.plate ? h('span', { class: 'plate' }, r.plate) : null,
        h('div', { class: 'sub' }, r.route ? `${r.route}` : (r.first_seen != null ? `first seen ${fmt(r.first_seen)}` : '')),
        r.clothing ? clothingLine(r.clothing) : null,
        r.score != null ? h('div', { class: 'sub' }, `${Math.round(r.score * 100)}% match`) : null));
    card.addEventListener('click', () => (r.kind === 'vehicle' ? openVehicle(r) : openSubject(r.camera_id, r.track_id)));
    el.rail.append(card);
  });
}

function micIcon() {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('width', '20'); s.setAttribute('height', '20'); s.setAttribute('aria-hidden', 'true');
  s.innerHTML = '<path fill="currentColor" d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-2.08A7 7 0 0 0 19 12h-2z"/>';
  return s;
}
