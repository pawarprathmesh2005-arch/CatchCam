// Watchlist + alerts: what to keep watching for, and what has turned up. Also the bell with a count in the header.
import { h, clear, api, toast } from './util.js';

const KINDS = {
  plate:       { label: 'Number plate', placeholder: 'MH12AB1234 (or the start, MH12)', needsValue: true },
  clothing:    { label: 'Clothing', placeholder: 'red jacket, or blue shirt and black pants', needsValue: true },
  gap:         { label: 'Unaccounted gap longer than', placeholder: 'seconds, e.g. 30', needsValue: true },
  camera_down: { label: 'Any camera goes down', placeholder: '', needsValue: false },
};

export function mountAlerts(host) {
  const kind = h('select', { class: 'input', id: 'w-kind', 'aria-label': 'What to watch for' }, Object.entries(KINDS).map(([k, v]) => h('option', { value: k }, v.label)));
  const value = h('input', { class: 'input', id: 'w-value', type: 'text', maxlength: '80', autocomplete: 'off', placeholder: KINDS.plate.placeholder, 'aria-label': 'Value' });
  const add = h('button', { class: 'btn primary', type: 'submit' }, 'Watch for this');
  const form = h('form', { class: 'watch-form' }, kind, value, add);
  const items = h('div', { class: 'chips', id: 'w-items' });
  const note = h('p', { class: 'note', id: 'w-note' });
  const log = h('div', { id: 'w-log', 'aria-live': 'polite' });
  const head = h('div', { class: 'acc-bar' }, h('h3', {}, 'Watchlist and alerts'), h('span', { class: 'grow' }));

  kind.addEventListener('change', () => {
    const k = KINDS[kind.value];
    value.hidden = !k.needsValue; value.placeholder = k.placeholder; if (!k.needsValue) value.value = '';
  });
  form.addEventListener('submit', async e => {
    e.preventDefault();
    add.disabled = true;
    try {
      const r = await api('/alerts/watch', { kind: kind.value, value: value.value }, { method: 'POST' });
      value.value = '';
      toast(r.existing_hits ? `Added. ${r.existing_hits} match${r.existing_hits > 1 ? 'es are' : ' is'} already in the footage (shown below).` : 'Added. You will be alerted when it turns up.');
      await refresh();
    } catch (err) { toast(err.message, 'error'); }
    finally { add.disabled = false; }
  });

  async function act(path, params, method = 'POST') {
    try { await api(path, params, { method }); await refresh(); } catch (err) { toast(err.message, 'error'); }
  }

  function draw(d) {
    clear(items);
    if (!d.items.length) items.append(h('p', { class: 'empty' }, 'Nothing on the watchlist yet. Add a plate, a clothing description, or a gap length above.'));
    d.items.forEach(i => items.append(h('span', { class: 'chip watch' }, i.label, i.alerts ? h('b', {}, String(i.alerts)) : null,
      h('button', { type: 'button', class: 'x', 'aria-label': `Stop watching: ${i.label}`, onClick: () => act(`/alerts/watch/${i.id}`, {}, 'DELETE') }, '×'))));
    const ch = Object.entries(d.notifications).filter(([, on]) => on).map(([n]) => n);
    note.textContent = ch.length ? `New alerts are also sent to: ${ch.join(', ')}.` : 'Alerts appear here and on the bell in the top bar. To also send them to Telegram or Slack, see README_WEB.md.';
    clear(head).append(h('h3', {}, 'Watchlist and alerts'), h('span', { class: 'grow' }),
      d.unacknowledged ? h('button', { class: 'btn small', type: 'button', onClick: () => act('/alerts/ack', { everything: true }) }, `Acknowledge all (${d.unacknowledged})`) : null);
    clear(log);
    if (!d.alerts.length) { log.append(h('p', { class: 'empty' }, 'No alerts so far.')); return; }
    log.append(h('ul', { class: 'list alerts' }, d.alerts.slice(0, 40).map(a => h('li', { class: a.acknowledged ? 'done' : 'new', 'data-kind': a.kind },
      h('div', { class: 'grow' }, h('div', { class: 'who' }, a.title, a.historical ? h('span', { class: 'pill' }, 'already in the footage') : null),
        h('div', { class: 'sub' }, `${a.created_text} · ${a.detail}`, a.notification ? ` · ${a.notification}` : '')),
      h('a', { class: 'btn small', href: a.link }, a.track_id != null ? 'Investigate' : 'Cameras'),
      a.acknowledged ? null : h('button', { class: 'btn small ghost', type: 'button', onClick: () => act('/alerts/ack', { alert_id: a.id }) }, 'Acknowledge')))));
  }

  async function refresh() {
    try { draw(await api('/alerts')); window.dispatchEvent(new CustomEvent('catchcam:alerts-changed')); }
    catch (e) { clear(log).append(h('p', { class: 'empty' }, e.message)); }
  }

  host.append(head, h('p', { class: 'muted', style: 'max-width:75ch' }, 'Tell CatchCam what to keep an eye out for. When new footage is processed and something matches, it appears here. Things already in the footage when you add an item are shown too, but marked.'),
    form, items, note, log);
  refresh();
  const timer = setInterval(() => { if (!host.isConnected) clearInterval(timer); else if (!host.closest('[hidden]')) refresh(); }, 8000);
}

// the bell in the header: a count of unacknowledged alerts; a short message when a new one arrives
export function startBell(link, badge) {
  let last = null;
  async function poll() {
    try {
      const n = (await api('/alerts/count')).unacknowledged;
      badge.textContent = n > 99 ? '99+' : String(n);
      badge.hidden = n === 0;
      link.setAttribute('aria-label', n ? `${n} unacknowledged alert${n > 1 ? 's' : ''}` : 'No new alerts');
      if (last !== null && n > last) toast(`${n - last} new alert${n - last > 1 ? 's' : ''}. Open Insights to see ${n - last > 1 ? 'them' : 'it'}.`, 'error');
      last = n;
    } catch { /* the status light in the header already says when the server is unreachable */ }
  }
  window.addEventListener('catchcam:alerts-changed', poll);
  poll();
  setInterval(poll, 8000);
}
