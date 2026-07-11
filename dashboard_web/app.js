/* PHAROS live dashboard — vanilla JS. Reads the SSE bridge (/events, /reference)
   and renders everything by hand on <canvas>; no charting framework, no build.
   All color comes from the CSS custom properties so light/dark stay in one place. */
'use strict';

const $ = (s) => document.querySelector(s);
const css = (name) => getComputedStyle(document.documentElement)
  .getPropertyValue(name).trim();
const COL = {
  ref: css('--ref'), cur: css('--cur'), phys: css('--phys'), pdm: css('--pdm'),
  grid: css('--grid'), muted: css('--muted'), ink2: css('--ink-2'),
  base: css('--baseline'),
};

let reference = {};                       // { physics:[...], pdm:{sys:[...]} }
const sparkHist = { phys: [], pdm: [] };  // rolling throughput samples
const SPARK_MAX = 120;

/* ---------- device-pixel-ratio aware canvas ---------- */
function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w * dpr || cv.height !== h * dpr) {
    cv.width = w * dpr; cv.height = h * dpr;
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/* ---------- overlaid log-x histogram (reference vs current) ---------- */
function drawHist(cv, ref, cur) {
  const { ctx, w, h } = fitCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 8, r: 8, t: 10, b: 22 };
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;

  const logv = (a) => a.filter((x) => x > 0).map((x) => Math.log10(x));
  const lref = logv(ref), lcur = logv(cur);
  const all = lref.concat(lcur);
  if (all.length < 2) { placeholder(ctx, w, h); return; }
  // robust range (0.5–99.5 pct) so tails don't crush the body
  const sorted = all.slice().sort((a, b) => a - b);
  const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
  let lo = q(0.005), hi = q(0.995);
  if (hi <= lo) hi = lo + 1;
  const NB = 34, binW = (hi - lo) / NB;
  const bins = (a) => {
    const c = new Array(NB).fill(0);
    for (const x of a) {
      let k = Math.floor((x - lo) / binW);
      if (k < 0 || k >= NB) continue; c[k]++;
    }
    const tot = a.length || 1;
    return c.map((v) => v / tot);          // density (fair shape overlay)
  };
  const bref = bins(lref), bcur = bins(lcur);
  const ymax = Math.max(...bref, ...bcur, 1e-6) * 1.15;

  // gridlines
  ctx.strokeStyle = COL.grid; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + plotH * (i / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotW, y); ctx.stroke();
  }
  const bx = (k) => pad.l + plotW * (k / NB);
  const by = (v) => pad.t + plotH * (1 - v / ymax);

  // reference: filled area
  drawBars(ctx, bref, bx, by, plotH + pad.t, binW, plotW, NB, COL.ref, 0.30, true);
  // current: outline
  drawBars(ctx, bcur, bx, by, plotH + pad.t, binW, plotW, NB, COL.cur, 0, false);

  // x ticks (log10 labels)
  ctx.fillStyle = COL.muted; ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const t = lo + (hi - lo) * (i / 4);
    ctx.fillText(t.toFixed(1), pad.l + plotW * (i / 4), h - 7);
  }
}

function drawBars(ctx, b, bx, by, baseY, binW, plotW, NB, color, fillAlpha, fill) {
  const barW = plotW / NB;
  ctx.lineWidth = 2; ctx.strokeStyle = color;
  if (fill) {
    ctx.fillStyle = hexA(color, fillAlpha);
    ctx.beginPath(); ctx.moveTo(bx(0), baseY);
    for (let k = 0; k < NB; k++) { ctx.lineTo(bx(k), by(b[k])); ctx.lineTo(bx(k + 1), by(b[k])); }
    ctx.lineTo(bx(NB), baseY); ctx.closePath(); ctx.fill();
  }
  ctx.beginPath();
  for (let k = 0; k < NB; k++) {
    const x0 = bx(k), x1 = bx(k + 1), y = by(b[k]);
    if (k === 0) ctx.moveTo(x0, y); else ctx.lineTo(x0, y);
    ctx.lineTo(x1, y);
  }
  ctx.stroke();
}

function placeholder(ctx, w, h) {
  ctx.fillStyle = COL.muted; ctx.font = '12px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('awaiting stream…', w / 2, h / 2);
}
function hexA(hex, a) {
  const n = parseInt(hex.replace('#', ''), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/* ---------- throughput sparklines ---------- */
function drawSpark(cv) {
  const { ctx, w, h } = fitCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 8, r: 8, t: 10, b: 16 };
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const series = [[sparkHist.phys, COL.phys], [sparkHist.pdm, COL.pdm]];
  const ymax = Math.max(1, ...sparkHist.phys, ...sparkHist.pdm) * 1.2;

  ctx.strokeStyle = COL.grid; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = pad.t + plotH * (i / 3);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + plotW, y); ctx.stroke();
  }
  for (const [data, color] of series) {
    if (data.length < 2) continue;
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad.l + plotW * (i / (SPARK_MAX - 1));
      const y = pad.t + plotH * (1 - v / ymax);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    // end dot
    const last = data[data.length - 1];
    const x = pad.l + plotW * ((data.length - 1) / (SPARK_MAX - 1));
    const y = pad.t + plotH * (1 - last / ymax);
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
  }
  ctx.fillStyle = COL.muted; ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'left'; ctx.fillText(ymax.toFixed(0) + ' ev/s', pad.l + 2, pad.t + 4);
}

/* ---------- kept/dropped bars ---------- */
function renderKeep(el, streams) {
  el.innerHTML = '';
  for (const [name, s] of streams) {
    const kept = s.kept || 0, dropped = s.dropped || 0, tot = kept + dropped;
    const pk = tot ? (kept / tot) * 100 : 0;
    const row = document.createElement('div'); row.className = 'kd';
    row.innerHTML =
      `<span class="name">${name}</span>
       <span class="track">
         <span class="kept" style="width:${pk}%"></span>
         <span class="dropped" style="width:${100 - pk}%"></span>
       </span>
       <span class="fig">${kept.toLocaleString()} kept · ${dropped.toLocaleString()} dropped</span>`;
    el.appendChild(row);
  }
}

/* ---------- drift feed ---------- */
let lastFeedKey = '';
function renderFeed(drift) {
  const feed = $('#feed');
  $('#drift-count').textContent = `${drift.n_total.toLocaleString()} evaluations`;
  const evs = drift.events || [];
  if (!evs.length) {
    feed.innerHTML = `<li class="feed-empty">no warn/alert drift events yet (${drift.n_total} evaluations seen)</li>`;
    return;
  }
  const key = evs.map((e) => e.detected_ts_ns || e.value).join('|');
  if (key === lastFeedKey) return;             // no change, skip repaint
  lastFeedKey = key;
  feed.innerHTML = '';
  for (const e of evs) {
    const li = document.createElement('li'); li.className = 'enter';
    li.innerHTML =
      `<span class="sev ${e.severity}">${e.severity.toUpperCase()}</span>
       <span class="meta">${e.stream} · ${e.metric}</span>
       <span class="val">${Number(e.value).toFixed(3)} <small style="color:var(--muted)">n=${e.window_n}</small></span>`;
    feed.appendChild(li);
  }
}

/* ---------- snapshot → DOM ---------- */
function setText(k, v) { const el = document.querySelector(`[data-k="${k}"]`); if (el) el.textContent = v; }
function pct(x) { return x == null ? '—' : (x * 100).toFixed(2) + '%'; }

function apply(snap) {
  const p = snap.physics, d = snap.pdm;
  setText('phys-rate', p.rate.toFixed(0));
  setText('pdm-rate', d.rate.toFixed(1));
  setText('phys-keep', pct(p.keep_rate));
  setText('pdm-keep', pct(d.keep_rate));
  setText('phys-keep-sub', `${p.kept} kept / ${p.dropped} dropped`);
  setText('pdm-keep-sub', `${d.kept} kept / ${d.dropped} dropped`);
  setText('reduction', p.keep_rate ? (1 / p.keep_rate).toFixed(0) : '—');
  setText('pdm-system', d.system ? d.system : '');

  sparkHist.phys.push(p.rate); if (sparkHist.phys.length > SPARK_MAX) sparkHist.phys.shift();
  sparkHist.pdm.push(d.rate); if (sparkHist.pdm.length > SPARK_MAX) sparkHist.pdm.shift();

  drawHist($('#hist-physics'), reference.physics || [], p.scores || []);
  const pref = (reference.pdm && d.system) ? reference.pdm[d.system] || [] : [];
  drawHist($('#hist-pdm'), pref, d.scores || []);
  drawSpark($('#spark'));
  renderKeep($('#keepbars'), [['physics', p], ['PDM', d]]);
  renderFeed(snap.drift);

  $('#foot').textContent =
    `consumed: physics ${p.n_total.toLocaleString()} · pdm ${d.n_total.toLocaleString()} · drift ${snap.drift.n_total.toLocaleString()}`;
}

/* ---------- SSE wiring with reconnect ---------- */
function setConn(state, text) {
  const c = $('#conn'); c.className = 'conn ' + state;
  $('#conn-text').textContent = text;
}
let lastSnap = null;
function connect() {
  const es = new EventSource('/events');
  es.onopen = () => setConn('live', 'live');
  es.onmessage = (m) => { try { lastSnap = JSON.parse(m.data); apply(lastSnap); } catch (e) {} };
  es.onerror = () => {
    setConn('down', 'reconnecting…');
    es.close(); setTimeout(connect, 2000);   // EventSource also auto-retries; belt & braces
  };
}

async function boot() {
  try { reference = await (await fetch('/reference')).json(); } catch (e) { reference = {}; }
  connect();
  window.addEventListener('resize', () => { if (lastSnap) apply(lastSnap); });
}
boot();
