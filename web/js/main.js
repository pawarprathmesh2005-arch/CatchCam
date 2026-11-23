// Router and header. Three screens: Investigate (home), Cameras, Insights.
// The address bar holds the place:  #investigate   #investigate/cam1   #investigate/cam1/5   #cameras   #insights
import { $, h, clear, api } from './util.js';
import * as investigate from './investigate.js';
import * as cameras from './cameras.js';
import * as insights from './insights.js';
import { startBell } from './alerts.js';

const VIEWS = { investigate, cameras, insights };
const mounted = new Set();
let current = null;

function parse() {
  const [v, ...rest] = location.hash.replace(/^#/, '').split('/');
  return { view: VIEWS[v] ? v : 'investigate', args: rest.map(decodeURIComponent) };
}

async function route() {
  const { view, args } = parse();
  if (current && current !== view) VIEWS[current].hide?.();
  Object.keys(VIEWS).forEach(name => {
    $(`#view-${name}`).hidden = name !== view;
    const link = $(`.tabs a[data-view="${name}"]`);
    if (name === view) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
  if (!mounted.has(view)) { VIEWS[view].mount($(`#view-${view}`)); mounted.add(view); }
  current = view;
  document.title = `${{ investigate: 'Investigate', cameras: 'Cameras', insights: 'Insights' }[view]} · CatchCam`;
  await VIEWS[view].show?.(...args);
}

// the little status light in the header: how many cameras, and whether any are down
async function status() {
  const box = $('#sys');
  try {
    const cams = (await api('/web/cameras')).cameras;
    const down = cams.filter(c => c.status === 'down').length;
    clear(box).append(h('i', { class: `led ${down ? 'down' : cams.length ? 'live' : ''}` }),
      cams.length ? `${cams.length} ${cams.length === 1 ? 'camera' : 'cameras'}${down ? `, ${down} down` : ''}` : 'no cameras yet');
  } catch {
    clear(box).append(h('i', { class: 'led down' }), 'server not reachable');
  }
}

window.addEventListener('hashchange', route);
startBell($('#bell'), $('#bell-n'));
window.addEventListener('catchcam:cameras-changed', status);
route();
status();
setInterval(status, 10000);
