// Cameras: see every camera's status, add recorded footage (with upload progress), connect a live stream,
// and run the processing pipeline (tracking -> appearance -> matching -> clothing) with a visible progress bar.
import { $, h, clear, api, fmt, toast, clockToSeconds } from './util.js';

const S = { mode: 'upload', cams: [], file: null, jobs: new Map(), poll: null, speed: 'fast', sens: 'normal', empty: 'skip' };
try { Object.assign(S, JSON.parse(localStorage.getItem('catchcam.process') || '{}')); } catch { /* ignore */ }
let el = {};

export function mount(root) {
  el = { form: h('section', { class: 'panel add' }), jobs: h('div', { class: 'camlist', id: 'jobs', style: 'margin-bottom:18px' }), list: h('div', { class: 'camlist', id: 'camlist' }) };
  root.append(h('div', { class: 'page-head' }, h('h2', {}, 'Cameras and footage'), h('p', {}, 'Add a recorded video or a live camera, then process it so CatchCam can find and follow people.')),
    el.form, processOptions(), el.jobs, el.list);
  renderForm();
}

export async function show() {
  await refresh();
  try {
    (await api('/web/jobs')).jobs.filter(j => j.state === 'running' || j.state === 'queued').forEach(j => watchJob(j));
  } catch { /* the list below still works */ }
}

export function hide() { /* jobs keep running on the server; polling continues quietly */ }

async function refresh() {
  try { S.cams = (await api('/web/cameras')).cameras; } catch (e) { toast(e.message, 'error'); S.cams = []; }
  renderList();
  window.dispatchEvent(new CustomEvent('catchcam:cameras-changed'));
}

// how Process runs: faster (looks at fewer frames) and/or more sensitive (also counts people it is less sure about)
function processOptions() {
  const pick = (key, label, opts) => {
    const box = h('div', { class: 'seg-ctl', role: 'group', 'aria-label': label }, h('span', { class: 'muted' }, label));
    opts.forEach(([v, text, tip]) => {
      const b = h('button', { class: 'btn small ghost', type: 'button', title: tip, 'data-opt': `${key}:${v}`, 'aria-pressed': String(S[key] === v) }, text);
      b.addEventListener('click', () => {
        S[key] = v; try { localStorage.setItem('catchcam.process', JSON.stringify({ speed: S.speed, sens: S.sens, empty: S.empty })); } catch { /* ignore */ }
        box.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      });
      box.append(b);
    });
    return box;
  };
  return h('section', { class: 'panel proc-opts' }, h('b', {}, 'When processing'),
    pick('speed', 'Speed', [['normal', 'Every frame', 'Slowest, most detailed'], ['fast', 'Faster (every 3rd frame)', 'About 3 times quicker; fine for people walking'], ['fastest', 'Fastest (every 5th, smaller)', 'About 5 to 8 times quicker; may miss quick or distant people']]),
    pick('empty', 'Empty footage', [['skip', 'Skip it (faster)', 'Only analyse the stretches where something moves'], ['all', 'Analyse everything', 'Slower; use if people were missed']]),
    h('button', { class: 'btn small', type: 'button', id: 'clean-all', title: 'Remove "people" that were only blips: seen for a fraction of a second, low confidence, not moving',
      onClick: cleanAll }, 'Remove false detections'),
    pick('sens', 'Sensitivity', [['normal', 'Normal', ''], ['high', 'High', 'Also counts people the detector is less sure about (dark, green night-vision or distant footage). May add some false people.']]));
}

async function cleanAll() {
  try {
    const r = await api('/web/clean', {}, { method: 'POST' });
    toast(r.removed ? `Removed ${r.removed} false detections from ${r.cameras.filter(c => c.removed).length} camera(s).` : 'Nothing to remove: no false detections found.', 'ok');
    window.dispatchEvent(new Event('catchcam:cameras-changed'));
    await refresh();
  } catch (e) { toast(e.message, 'error'); }
}

// ----------------------------------------------------------------------------- add footage
function renderForm() {
  const name = h('input', { class: 'input', id: 'cam-name', placeholder: 'cam1', value: `cam${S.cams.length + 1}`, autocomplete: 'off', maxlength: '40' });
  const modes = h('div', { class: 'modes', role: 'tablist' }, [['upload', 'Upload a video'], ['stream', 'Live camera']].map(([k, label]) => {
    const b = h('button', { type: 'button', role: 'tab', 'aria-selected': String(S.mode === k) }, label);
    b.addEventListener('click', () => { S.mode = k; renderForm(); });
    return b;
  }));
  const body = h('div', { class: 'add' });
  if (S.mode === 'upload') {
    const input = h('input', { type: 'file', accept: 'video/*,.mkv,.avi', class: 'sr', id: 'file' });
    const drop = h('label', { class: 'drop', for: 'file', id: 'drop' }, h('strong', {}, S.file ? S.file.name : 'Choose a video file'),
      h('div', {}, S.file ? `${(S.file.size / 1e6).toFixed(1)} MB: change it by choosing another` : 'or drop it here · mp4, mov, webm, avi or mkv'));
    input.addEventListener('change', () => { S.file = input.files[0] || null; renderForm(); });
    ['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) { S.file = f; renderForm(); } });
    const auto = h('input', { type: 'checkbox', id: 'auto', checked: true });
    const offset = h('input', { class: 'input', id: 'offset', type: 'number', step: '0.1', value: '0' });
    const go = h('button', { class: 'btn primary', type: 'button', id: 'go' }, 'Upload');
    const prog = h('div', { class: 'bar-track', hidden: true }, h('div', { class: 'bar-fill', id: 'upfill' }));
    go.addEventListener('click', () => upload({ name, auto, offset, go, prog }));
    body.append(h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Camera name'), name), h('div', {}, input, drop)),
      h('label', { style: 'display:flex;gap:8px;align-items:center' }, auto, 'Process it straight after uploading (finds the people in it)'),
      h('details', { class: 'adv' }, h('summary', {}, 'Advanced'), h('label', { class: 'field' }, h('span', {}, 'This video starts at (seconds on the shared clock)'), offset),
        h('p', { class: 'muted', style: 'font-size:13px;margin-top:6px;max-width:60ch' }, 'Leave at 0 if all your cameras were recorded at the same time and start together. Use it when this camera started later.')),
      h('div', { style: 'display:flex;gap:12px;align-items:center' }, go, h('div', { style: 'flex:1' }, prog)));
  } else {
    const url = h('input', { class: 'input', id: 'stream-url', placeholder: 'rtsp://user:password@192.168.1.20:554/stream   or   0 for a webcam', autocomplete: 'off' });
    const go = h('button', { class: 'btn primary', type: 'button', id: 'go-stream' }, 'Connect');
    go.addEventListener('click', async () => {
      go.disabled = true;
      try { await api('/web/streams', { camera_id: name.value, url: url.value.trim() }, { method: 'POST' }); toast(`${name.value} connected.`, 'ok'); await refresh(); }
      catch (e) { toast(e.message, 'error'); }
      go.disabled = false;
    });
    body.append(h('div', { class: 'form-row' }, h('label', { class: 'field' }, h('span', {}, 'Camera name'), name), h('label', { class: 'field' }, h('span', {}, 'Camera address'), url)),
      h('p', { class: 'muted', style: 'font-size:13px;max-width:70ch' }, 'Most IP cameras give an RTSP address in their app or settings. A live camera is shown on the Investigate screen. To also track people live, run realtime_track.py for it; the status light turns green when it reports in.'),
      go);
  }
  clear(el.form).append(modes, body);
}

function upload({ name, auto, offset, go, prog }) {
  if (!S.file) { toast('Choose a video file first.', 'error'); return; }
  const cam = name.value.trim();
  if (!cam) { toast('Give the camera a name.', 'error'); return; }
  if (S.cams.some(c => c.camera_id === cam && c.has_video) && !confirm(`${cam} already has footage. Replace it?`)) return;
  const xhr = new XMLHttpRequest();
  const url = `/web/upload?camera_id=${encodeURIComponent(cam)}&filename=${encodeURIComponent(S.file.name)}`;
  const fill = $('#upfill');
  prog.hidden = false; go.disabled = true; go.textContent = 'Uploading…';
  xhr.upload.onprogress = e => { if (e.lengthComputable) fill.style.width = `${(e.loaded / e.total) * 100}%`; };
  xhr.onerror = () => { reset(); toast("Can't reach the CatchCam server. Is the API running?", 'error'); };
  xhr.onload = async () => {
    reset();
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
    if (xhr.status >= 300) { toast(typeof body.detail === 'string' ? body.detail : `Upload failed (${xhr.status}).`, 'error'); return; }
    toast(`${body.camera_id} uploaded (${body.size_mb} MB).`, 'ok');
    S.file = null; renderForm();
    await refresh();
    if (auto.checked) startProcess(body.camera_id, Number(offset.value) || 0);
  };
  function reset() { go.disabled = false; go.textContent = 'Upload'; prog.hidden = true; fill.style.width = '0'; }
  xhr.open('POST', url);
  xhr.send(S.file);
}

// ----------------------------------------------------------------------------- processing jobs
async function startProcess(cam, offset = 0) {
  try { watchJob(await api('/web/process', { camera_id: cam, time_offset: offset, speed: S.speed, sensitivity: S.sens, skip_empty: S.empty === 'skip' ? 1 : 0 }, { method: 'POST' })); }
  catch (e) { toast(e.message, 'error'); }
}

async function startConvert(cam) {
  try { watchJob(await api('/web/convert', { camera_id: cam }, { method: 'POST' })); }
  catch (e) { toast(e.message, 'error'); }
}

function watchJob(job) {
  const box = h('div', { class: 'job', id: `job-${job.id}` });
  el.jobs.prepend(box);
  const draw = j => {
    clear(box).className = `job${j.state === 'error' ? ' error' : ''}`;
    box.append(h('div', { class: 'row' }, h('b', {}, `${j.camera_id}: ${j.kind === 'convert' ? 'converting' : 'processing'}`), h('span', { class: 'muted', style: 'flex:1' },
      j.state === 'done' ? 'Finished' : j.state === 'error' ? 'Stopped' : j.stage), h('span', { class: 'badge' }, `${j.percent}%`)),
    h('div', { class: 'bar-track' }, h('div', { class: 'bar-fill', style: `width:${j.percent}%` })),
    j.error ? h('p', { style: 'color:var(--bad)' }, j.error) : null,
    j.notes.length ? h('ul', {}, j.notes.map(n => h('li', {}, n))) : null);
  };
  draw(job);
  const timer = setInterval(async () => {
    let j;
    try { j = await api(`/web/jobs/${job.id}`); } catch { return; }
    draw(j);
    if (j.state === 'done' || j.state === 'error') {
      clearInterval(timer);
      if (j.state === 'done') toast(`${j.camera_id} is ready.`, 'ok');
      refresh();
    }
  }, 1000);
}

async function removeCamera(c) {
  if (!confirm(`Delete ${c.camera_id}?\n\nThis removes its video and everything CatchCam found in it (people, clothing, matches). It cannot be undone.`)) return;
  try {
    await api(`/web/cameras/${encodeURIComponent(c.camera_id)}`, {}, { method: 'DELETE' });
    toast(`${c.camera_id} deleted.`, 'ok');
    const n = $('#cam-name'); if (n && /^cam\d+$/.test(n.value)) n.value = `cam${Math.max(1, S.cams.length)}`;
    await refresh();
  } catch (e) { toast(e.message, 'error'); }
}

// "When did this video start?" Type the clock shown on the camera's picture at the first frame, e.g. 12:38:53.
function startTimeControl(c) {
  const input = h('input', { class: 'input clock', type: 'text', inputmode: 'numeric', placeholder: 'hh:mm:ss', 'aria-label': `Start time of ${c.camera_id}`,
    value: c.offset ? fmt(c.offset) : '', size: '9' });
  const save = async () => {
    const secs = clockToSeconds(input.value);
    if (secs == null) { toast('Type the time as hh:mm:ss, for example 12:38:53.', 'error'); return; }
    try {
      await api(`/web/cameras/${encodeURIComponent(c.camera_id)}/start`, { seconds: secs }, { method: 'POST' });
      toast(`${c.camera_id} now starts at ${fmt(secs)}.`, 'ok');
      window.dispatchEvent(new Event('catchcam:cameras-changed'));
      await refresh();
    } catch (e) { toast(e.message, 'error'); }
  };
  input.addEventListener('keydown', e => { if (e.key === 'Enter') save(); });
  return h('div', { class: 'startrow', title: 'The clock shown on this camera at the first frame of the video. Lines cameras up so journeys make sense.' },
    h('span', { class: 'muted' }, 'Starts at'), input, h('button', { class: 'btn small ghost', type: 'button', onClick: save }, 'Set'));
}

// ----------------------------------------------------------------------------- the list
function renderList() {
  clear(el.list);
  if (!S.cams.length) { el.list.append(h('p', { class: 'empty' }, 'No cameras yet. Add your first video above.')); return; }
  S.cams.forEach(c => {
    const shot = c.has_video || c.is_stream
      ? h('img', { class: 'shot', src: `/web/frame?camera_id=${encodeURIComponent(c.camera_id)}&t=${(c.first_seen ?? 0) + 1}&width=320`, alt: '', loading: 'lazy' })
      : h('div', { class: 'shot nofeed', style: 'position:relative' }, 'no footage');
    shot.addEventListener?.('error', () => shot.replaceWith(h('div', { class: 'shot nofeed', style: 'position:relative' }, 'no picture')));
    const statusText = { live: 'live, reporting', down: `DOWN (no update for ${Math.round(c.last_update_seconds_ago ?? 0)}s)`, recorded: 'recorded video', stream: 'live stream', no_footage: 'no footage' }[c.status] || c.status;
    const meta = [c.num_tracks ? `${c.num_tracks} ${c.num_tracks === 1 ? 'person' : 'people'} found` : 'not processed yet', c.duration ? `${fmt(c.duration)} long` : null,
      c.has_video && !c.plays_in_browser ? 'needs conversion to play smoothly' : null].filter(Boolean).join(' · ');
    const acts = h('div', { class: 'acts' });
    if (c.has_video || c.is_stream) acts.append(h('a', { class: 'btn small', href: `#investigate/${encodeURIComponent(c.camera_id)}` }, 'Watch'));
    if (c.has_video && !c.is_stream) acts.append(h('button', { class: 'btn small', type: 'button', onClick: () => startProcess(c.camera_id, c.offset || 0) }, c.num_tracks ? 'Process again' : 'Process'));
    if (c.can_convert) acts.append(h('button', { class: 'btn small ghost', type: 'button', onClick: () => startConvert(c.camera_id) }, 'Convert for browser'));
    acts.append(h('button', { class: 'btn small ghost danger', type: 'button', onClick: () => removeCamera(c) }, 'Delete'));
    el.list.append(h('div', { class: 'camrow' }, shot, h('div', {}, h('div', { class: 'who', style: 'display:flex;gap:8px;align-items:center' }, h('i', { class: `led ${c.status}` }), c.camera_id),
      h('div', { class: 'sub' }, statusText), h('div', { class: 'sub' }, meta), c.has_video && !c.is_stream ? startTimeControl(c) : null), acts));
  });
}
