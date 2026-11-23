// "Accuracy check" panel (inside Insights): label who is who by watching the footage, then see how often CatchCam got it right.
import { h, clear, api, fmt, toast } from './util.js';

const pct = x => (x == null ? 'n/a' : `${(x * 100).toFixed(1)}%`);

export function mountAccuracy(host) {
  const S = { tracks: [], labels: new Map(), saveTimer: null, saving: null };
  const keyOf = t => `${t.camera_id}|${t.track_id}`;
  const counter = h('span', { class: 'muted', id: 'acc-count' });
  const savedNote = h('span', { class: 'muted', id: 'acc-saved' });
  const grid = h('div', { class: 'lab-grid', id: 'acc-grid' });
  const results = h('div', { id: 'acc-results', 'aria-live': 'polite' });
  const people = h('datalist', { id: 'acc-people' });
  const runBtn = h('button', { class: 'btn primary', type: 'button', id: 'acc-run' }, 'Run the check');

  const refreshCounts = () => {
    const n = [...S.labels.values()].filter(Boolean).length;
    counter.textContent = `${n} of ${S.tracks.length} tracks labelled`;
    clear(people).append(...[...new Set(S.labels.values())].filter(Boolean).sort().map(p => h('option', { value: p })));
  };

  function scheduleSave() {
    savedNote.textContent = 'Saving…';
    clearTimeout(S.saveTimer);
    S.saveTimer = setTimeout(saveNow, 700);
  }

  function saveNow() {
    clearTimeout(S.saveTimer);
    const rows = S.tracks.map(t => ({ camera_id: t.camera_id, track_id: t.track_id, person: S.labels.get(keyOf(t)) || '' }));
    S.saving = api('/eval/labels', {}, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ labels: rows }) })
      .then(r => { savedNote.textContent = `Saved (${r.saved} labelled)`; })
      .catch(e => { savedNote.textContent = ''; toast(e.message, 'error'); throw e; });
    return S.saving;
  }

  function card(t) {
    const input = h('input', { class: 'input', type: 'text', list: 'acc-people', maxlength: '40', placeholder: 'Who is this?', 'aria-label': `Person for ${t.camera_id} track ${t.track_id}`, value: S.labels.get(keyOf(t)) || '' });
    input.addEventListener('input', () => { S.labels.set(keyOf(t), input.value.trim()); refreshCounts(); scheduleSave(); });
    return h('div', { class: 'lab' },
      h('img', { class: 'thumb', src: t.thumb_url, alt: '', loading: 'lazy', onError: e => { e.target.replaceWith(h('div', { class: 'thumb blank' }, 'no picture')); } }),
      h('div', { class: 'lab-body' }, h('div', { class: 'who' }, `${t.camera_id} · #${t.track_id}`), h('div', { class: 'sub' }, `${fmt(t.first_seen)}–${fmt(t.last_seen)}`), input),
      h('a', { class: 'btn small ghost', href: `#investigate/${encodeURIComponent(t.camera_id)}/${t.track_id}`, title: 'Open this person in Investigate to check against the video' }, 'Watch'));
  }

  function drawGrid() {
    clear(grid);
    if (!S.tracks.length) { grid.append(h('p', { class: 'empty' }, 'No tracks yet. Add footage in Cameras first.')); return; }
    grid.append(...S.tracks.map(card));
    refreshCounts();
  }

  function guess() {
    let filled = 0;
    S.tracks.forEach(t => { if (!S.labels.get(keyOf(t))) { S.labels.set(keyOf(t), `Person ${t.predicted_group}`); filled++; } });
    drawGrid(); scheduleSave();
    toast(filled ? `Filled ${filled} labels from CatchCam's own grouping. Check each one against the video before you trust the score.` : 'Everything is already labelled.');
  }

  function clearAll() {
    if (![...S.labels.values()].some(Boolean) || !confirm('Remove all labels?')) return;
    S.labels.clear(); drawGrid(); scheduleSave();
  }

  async function run() {
    runBtn.disabled = true; runBtn.textContent = 'Checking…';
    try {
      await saveNow();
      showResults(await api('/eval/run'));
    } catch (e) { toast(e.message, 'error'); }
    finally { runBtn.disabled = false; runBtn.textContent = 'Run the check'; }
  }

  function stat(value, label, note) { return h('div', { class: 'stat' }, h('b', {}, value), h('span', {}, label), note ? h('em', {}, note) : null); }

  function showResults(r) {
    clear(results);
    if (r.status !== 'ok') { results.append(h('p', { class: 'empty' }, r.message)); return; }
    const p = r.pairs, k = r.ranking;
    results.append(h('h4', {}, 'Result'),
      h('p', { class: 'muted' }, `Scored on ${r.labelled_tracks} labelled tracks of ${r.people} people across ${r.cameras.join(', ')}.`),
      h('div', { class: 'stats' },
        stat(pct(r.precision), 'Precision', `${p.correct_matches} right, ${p.false_matches} wrong`),
        stat(pct(r.recall), 'Recall', `${p.missed_matches} missed`),
        stat(pct(r.f1), 'F1'),
        stat(`${r.journeys.fully_correct} of ${r.journeys.with_journey}`, 'journeys rebuilt exactly'),
        k ? stat(pct(k.rank1), 'Rank-1', `${k.queries} queries`) : null,
        k ? stat(pct(k.map), 'mAP') : null));
    const sections = [];
    const bad = r.journeys.details.filter(j => !j.correct);
    if (bad.length) sections.push(h('div', {}, h('h4', {}, 'Journeys that came out wrong'), h('ul', { class: 'list' }, bad.map(j => h('li', {}, h('div', { class: 'grow' },
      h('div', { class: 'who' }, `${j.person}: ${j.tracks.join(' → ')}`),
      h('div', { class: 'sub' }, [j.missing.length ? `Not linked: ${j.missing.join(', ')}.` : '', j.extra.length ? `Wrongly linked with: ${j.extra.join(', ')}.` : ''].filter(Boolean).join(' '))))))));
    if (r.false_match_list.length) sections.push(h('div', {}, h('h4', {}, 'Different people linked together'), h('ul', { class: 'list' },
      r.false_match_list.map(m => h('li', {}, h('div', { class: 'grow' }, h('div', { class: 'who' }, `${m.a} (${m.a_person}) ↔ ${m.b} (${m.b_person})`)))))));
    if (r.fragmentation.length) sections.push(h('div', {}, h('h4', {}, 'Same person, several tracks on one camera'), h('ul', { class: 'list' },
      r.fragmentation.map(f => h('li', {}, h('div', { class: 'grow' }, h('div', { class: 'who' }, `${f.person} on ${f.camera_id}`), h('div', { class: 'sub' }, `Tracks ${f.tracks.join(', ')}: the tracker lost and re-found them.`)))))));
    if (r.threshold_sweep.length) sections.push(h('div', {}, h('h4', {}, 'Would a different similarity cut-off help?'),
      h('table', { class: 'sweep' }, h('thead', {}, h('tr', {}, ['Cut-off', 'Precision', 'Recall', 'F1', 'Wrong', 'Missed'].map(c => h('th', {}, c)))),
        h('tbody', {}, r.threshold_sweep.map(s => h('tr', {}, h('td', {}, s.threshold.toFixed(2)), h('td', {}, pct(s.precision)), h('td', {}, pct(s.recall)), h('td', {}, pct(s.f1)), h('td', {}, s.false_matches), h('td', {}, s.missed)))))));
    if (sections.length) results.append(h('div', { class: 'two' }, sections));
    r.notes.forEach(n => results.append(h('p', { class: 'note' }, n)));
    results.append(h('p', {}, h('a', { class: 'btn small', href: '/eval/report.md', download: 'catchcam_evaluation.md' }, 'Download write-up (.md)'),
      h('span', { class: 'muted', style: 'margin-left:10px;font-size:13px' }, 'Frame-by-frame tracking accuracy (IDF1) is not measured: it needs a box labelled on every frame.')));
  }

  host.append(h('h3', {}, 'Accuracy check'),
    h('p', { class: 'muted', style: 'max-width:75ch' }, 'How often is CatchCam right? Tell it who is who by watching the footage, then it scores its own cross-camera matching against your answers. Only the tracks you label are scored.'),
    h('div', { class: 'acc-bar' }, counter, savedNote,
      h('span', { class: 'grow' }),
      h('button', { class: 'btn small', type: 'button', id: 'acc-guess', onClick: guess, title: 'Fills empty boxes with CatchCam\'s own grouping' }, 'Start from CatchCam\'s guess'),
      h('button', { class: 'btn small ghost', type: 'button', onClick: clearAll }, 'Clear'),
      runBtn),
    h('p', { class: 'note' }, 'Type the same name (P1, Asha…) for every track of one person. Use Watch to check against the video. Starting from the guess and not checking inflates the score.'),
    people, grid, results);
  runBtn.addEventListener('click', run);

  (async () => {
    try {
      const d = await api('/eval/tracks');
      S.tracks = d.tracks;
      d.tracks.forEach(t => { if (t.label) S.labels.set(keyOf(t), t.label); });
      drawGrid();
    } catch (e) { grid.append(h('p', { class: 'empty' }, e.message)); }
  })();
}
