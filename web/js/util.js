// Small helpers shared by every screen. Text is always inserted with textContent (never innerHTML),
// so camera names, plates or error messages from the server can't inject markup.

export const $ = (sel, root = document) => root.querySelector(sel);

export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2).toLowerCase(), v);
    else if (v === true) e.setAttribute(k, '');
    else e.setAttribute(k, v);
  }
  append(e, kids);
  return e;
}

export function svg(tag, attrs = {}, ...kids) {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null && v !== false) e.setAttribute(k, v);
  append(e, kids);
  return e;
}

function append(e, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

export async function api(path, params = {}, opts = {}) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== '') q.set(k, v);
  const url = path + (q.toString() ? '?' + q : '');
  let res;
  try { res = await fetch(url, opts); }
  catch { throw new Error("Can't reach the CatchCam server. Is the API running?"); }
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).detail; } catch { /* not JSON */ }
    const err = new Error(typeof detail === 'string' && detail ? detail : `The server said ${res.status}.`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// seconds on the shared clock -> 1:05 (or 1:05.3 when precise)
export function fmt(t, precise = false) {
  if (t == null || Number.isNaN(t)) return '–';
  const neg = t < 0; t = Math.abs(t);
  const m = Math.floor(t / 60), s = t - m * 60;
  const sec = precise ? s.toFixed(1).padStart(4, '0') : String(Math.floor(s)).padStart(2, '0');
  return `${neg ? '-' : ''}${m}:${sec}`;
}

export const COLOR_HEX = { black: '#1b1d22', white: '#f1f2f4', gray: '#8b9099', silver: '#c4c8cf', red: '#d9453d', orange: '#e8873a',
  yellow: '#e8c93a', green: '#4aa86a', blue: '#3f78d6', purple: '#8d5bd0', brown: '#8a5a3b' };

export function clothingLine(c) {
  if (!c || (!c.upper && !c.lower)) return null;
  const part = (name, col) => col ? h('span', { class: 'cloth' }, h('i', { class: 'swatch', style: `background:${COLOR_HEX[col] || '#777'}` }), `${col} ${name}`) : null;
  return h('span', {}, part('top', c.upper), part('bottoms', c.lower));
}

export function toast(msg, kind = '') {
  const box = $('#toasts');
  const t = h('div', { class: `toast ${kind}` }, msg);
  box.append(t);
  setTimeout(() => t.remove(), kind === 'error' ? 8000 : 4000);
}

export function speak(text) {
  if (!('speechSynthesis' in window) || !text) { toast("This browser can't read aloud."); return; }
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1;
  window.speechSynthesis.speak(u);
}

export function listen(onText, onState) {
  const R = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!R) { toast('Voice input works in Chrome and Edge. In this browser, type your question instead.'); return null; }
  const r = new R();
  r.lang = 'en-IN'; r.interimResults = false; r.maxAlternatives = 1;
  r.onstart = () => onState(true);
  r.onend = () => onState(false);
  r.onerror = e => { onState(false); if (e.error === 'not-allowed') toast('Microphone access is blocked. Allow it in the address bar to use voice.', 'error'); };
  r.onresult = e => onText(e.results[0][0].transcript);
  r.start();
  return r;
}
