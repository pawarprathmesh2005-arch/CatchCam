// Video surfaces: a camera shown as a <video> (seekable) when the browser can play it, otherwise as a server-drawn
// stream (works for any codec and for live feeds). Person boxes are drawn on a canvas over the video.
import { h, api } from './util.js';

const SUBJECT = '#ffd23f';

export function makeSurface(cam, o = {}) {
  const { controls = false, loop = true, width = 640, startAt = null, track = null, forceStream = false, showOthers = true, still = false } = o;
  const root = h('div', { class: 'surface' });
  const canvas = h('canvas', { class: 'overlay' });
  const state = { kind: null, media: null, dead: false, track, showOthers, streamT0: 0, streamShared0: 0, overlay: null };
  const offset = cam.offset || 0;

  function mountVideo() {
    state.kind = 'video';
    const v = h('video', { src: cam.video_url, playsinline: true, preload: 'auto' });
    v.muted = true; v.loop = !!loop; v.controls = !!controls;
    v.addEventListener('error', () => { if (!state.dead && state.kind === 'video') mountStream(startAt == null ? null : currentShared()); });
    if (startAt != null) v.addEventListener('loadedmetadata', () => { v.currentTime = Math.max(startAt - offset, 0); }, { once: true });
    state.media = v;
    root.replaceChildren(v, canvas);
    state.overlay = attachOverlay(state, canvas, cam);
  }

  function mountStream(at) {
    state.kind = 'stream';
    state.overlay?.destroy(); state.overlay = null;
    const p = new URLSearchParams({ width: String(width), loop: loop ? '1' : '0' });
    const from = at ?? startAt;
    if (from != null) p.set('start', String(Math.max(from, 0)));
    if (state.track != null) p.set('track_id', String(state.track));
    const img = h('img', { src: `${cam.stream_url}?${p}`, alt: `${cam.camera_id} live view` });
    img.addEventListener('error', () => { if (!state.dead) root.replaceChildren(h('div', { class: 'nofeed' }, `Can't show ${cam.camera_id} right now.`)); });
    state.streamT0 = performance.now(); state.streamShared0 = from ?? 0;
    state.media = img;
    root.replaceChildren(img);
  }

  function currentShared() {
    if (state.kind === 'video') return state.media.currentTime + offset;
    return state.streamShared0 + (performance.now() - state.streamT0) / 1000;
  }

  const hasFootage = !!(cam.video_url || cam.stream_url);

  // A picture instead of a playing video. Browsers allow only ~6 open connections to a server, and every playing video or
  // live stream keeps one open, so a wall of 10 cameras must not play them all at once (the last ones stay black and the
  // rest of the page, search included, queues behind them).
  function mountStill() {
    state.kind = 'still';
    state.overlay?.destroy(); state.overlay = null;
    const t = cam.first_seen != null ? cam.first_seen + 1 : 1;
    const img = h('img', { src: `/web/frame?camera_id=${encodeURIComponent(cam.camera_id)}&t=${t}&width=${width}`, alt: `${cam.camera_id} (picture)` });
    img.addEventListener('error', () => { if (!state.dead) root.replaceChildren(h('div', { class: 'nofeed' }, `No picture for ${cam.camera_id} yet.`)); });
    state.media = img;
    root.replaceChildren(img);
  }

  function mountLive() {
    if (cam.plays_in_browser && cam.video_url && !cam.is_stream && !forceStream) mountVideo(); else mountStream();
  }

  if (!hasFootage) root.append(h('div', { class: 'nofeed' }, `No footage for ${cam.camera_id} yet.`));
  else if (still) mountStill();
  else mountLive();

  return {
    root, cam,
    get kind() { return state.kind; },
    get ended() { return state.kind === 'video' && !!state.media.ended; },
    currentShared,
    seek(t) {
      if (state.kind === 'video') state.media.currentTime = Math.max(t - offset, 0);
      else if (state.kind === 'stream') mountStream(t);
    },
    play() { if (state.kind === 'video') state.media.play().catch(() => {}); },
    get live() { return state.kind === 'video' || state.kind === 'stream'; },
    goLive() { if (state.kind === 'still') { mountLive(); this.play(); } },
    goStill() {
      if (!hasFootage || state.kind === 'still') return;
      const m = state.media;
      if (state.kind === 'video') { m.pause(); m.removeAttribute('src'); m.load(); } else if (m) m.removeAttribute('src');   // closes the connection
      mountStill();
    },
    pause() { if (state.kind === 'video') state.media.pause(); },
    setRate(r) { if (state.kind === 'video') state.media.playbackRate = r; },
    setSubject(t, others = true) { state.track = t; state.showOthers = others; },
    destroy() {
      state.dead = true; state.overlay?.destroy();
      const m = state.media;
      if (m && state.kind === 'video') { m.pause(); m.removeAttribute('src'); m.load(); }
      else if (m) m.removeAttribute('src');
    },
  };
}

function attachOverlay(state, canvas, cam) {
  let rows = [], win = [0, -1], loading = false, raf = 0, dead = false;
  const offset = cam.offset || 0;
  const ctx = canvas.getContext('2d');

  async function load(t) {
    if (loading) return;
    loading = true;
    const start = Math.max(Math.floor(t) - 5, 0), end = start + 60;
    try {
      const r = await api('/web/boxes', { camera_id: cam.camera_id, start, end });
      rows = r.rows; win = [start, end];
    } catch { win = [start, end]; rows = []; }
    loading = false;
  }

  function lowerBound(t) {
    let lo = 0, hi = rows.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (rows[mid][0] < t) lo = mid + 1; else hi = mid; }
    return lo;
  }

  function frame() {
    if (dead) return;
    raf = requestAnimationFrame(frame);
    const v = state.media;
    if (!v || state.kind !== 'video' || !v.videoWidth) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
    const t = v.currentTime + offset;
    if (!loading && (t < win[0] || t > win[1] - 6)) load(t);
    const dpr = window.devicePixelRatio || 1, cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) { canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    const s = Math.min(cw / v.videoWidth, ch / v.videoHeight), ox = (cw - v.videoWidth * s) / 2, oy = (ch - v.videoHeight * s) / 2;
    const best = new Map();
    for (let i = lowerBound(t - 0.35); i < rows.length && rows[i][0] <= t + 0.35; i++) {
      const r = rows[i], cur = best.get(r[1]);
      if (!cur || Math.abs(r[0] - t) < Math.abs(cur[0] - t)) best.set(r[1], r);
    }
    ctx.font = '600 12px system-ui, sans-serif';
    for (const [id, r] of best) {
      const mine = state.track != null && id === state.track;
      if (state.track != null && !mine && !state.showOthers) continue;
      const x = ox + r[2] * s, y = oy + r[3] * s, w = (r[4] - r[2]) * s, hh = (r[5] - r[3]) * s;
      ctx.lineWidth = mine ? 3 : 1.5;
      ctx.strokeStyle = mine ? SUBJECT : (state.track != null ? 'rgba(160,180,210,.5)' : 'rgba(190,205,230,.85)');
      ctx.strokeRect(x, y, w, hh);
      if (mine || state.track == null) {
        const label = `#${id}`, tw = ctx.measureText(label).width + 10;
        ctx.fillStyle = mine ? SUBJECT : 'rgba(190,205,230,.85)';
        ctx.fillRect(x - 1, y - 18, tw, 18);
        ctx.fillStyle = '#17130a';
        ctx.fillText(label, x + 4, y - 5);
      }
    }
  }
  raf = requestAnimationFrame(frame);
  return { destroy() { dead = true; cancelAnimationFrame(raf); } };
}
