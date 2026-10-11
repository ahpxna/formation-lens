"use strict";
const $ = (s) => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const N_TEAM = 11, BALL = 22, N_AGENTS = 23;
const LINK_M = 16, TRAIL_S = 6;

const state = {
  doc: null, view: "half1", sample: 0, t: 0, playing: false, speed: 16,
  layers: { links: true, hull: true, line: true, trails: false, formation: false, compare: false },
  windowMin: 5, metric: "width", gk: [0, N_TEAM], h1Avg: null, metrics: null, lastPanel: 0,
};

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

// ---------- geometry ----------
// Attacking direction along x; the model's convention is team 0 -> +x, video exports keep the camera's view.
const dir = (k) => (state.doc && state.doc.attackDir ? state.doc.attackDir[k] : k === 0 ? 1 : -1);
const halfLen = () => state.doc.pitch[0] / 2;

function hull(pts) {
  if (pts.length < 3) return pts.slice();
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
const area = (h) => Math.abs(h.reduce((s, a, i) => { const b = h[(i + 1) % h.length]; return s + a[0] * b[1] - b[0] * a[1]; }, 0)) / 2;

function outfield(xy, k) {
  const pts = [];
  for (let s = k * N_TEAM; s < (k + 1) * N_TEAM; s++) {
    if (s === state.gk[k]) continue;
    const x = xy[2 * s], y = xy[2 * s + 1];
    if (Number.isFinite(x)) pts.push([x, y, s]);
  }
  return pts;
}

function teamMetrics(xy, k) {
  const pts = outfield(xy, k);
  if (pts.length < 4) return null;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const d = dir(k), L = halfLen();
  const deep = pts.map((p) => d * p[0]).sort((a, b) => a - b).slice(0, 4);
  const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
  return {
    width: Math.max(...ys) - Math.min(...ys),
    depth: Math.max(...xs) - Math.min(...xs),
    area: area(hull(pts)),
    defLine: L + deep.reduce((a, b) => a + b, 0) / deep.length,
    height: L + d * cx,
    n: pts.length,
  };
}

// Optimal 1-D split of outfield players (sorted along the attacking axis) into 3 or 4 lines, minimising
// within-line variance, with a back line of 3-5; 4 lines only when they explain clearly more than 3.
function formationLabel(avg, k) {
  const xs = outfield(avg, k).map((p) => dir(k) * p[0]).sort((a, b) => a - b);
  if (xs.length < 8) return "–";
  const sse = (i, j) => { const s = xs.slice(i, j), m = s.reduce((a, b) => a + b, 0) / s.length; return s.reduce((a, b) => a + (b - m) ** 2, 0); };
  const best = (nLines) => {
    let top = { cost: Infinity, sizes: null };
    const rec = (start, left, sizes, cost) => {
      if (left === 1) {
        const all = sizes.concat(xs.length - start);
        if (all[0] >= 3 && all[0] <= 5 && cost + sse(start, xs.length) < top.cost) top = { cost: cost + sse(start, xs.length), sizes: all };
        return;
      }
      for (let end = start + 1; end <= xs.length - (left - 1); end++) rec(end, left - 1, sizes.concat(end - start), cost + sse(start, end));
    };
    rec(0, nLines, [], 0);
    return top;
  };
  const three = best(3), four = best(4);
  const pick = four.sizes && four.cost < 0.6 * three.cost ? four : three;
  return pick.sizes ? pick.sizes.join("-") : "–";
}

// ---------- series ----------
function series(view = state.view) {
  const d = state.doc;
  if (view === "pred") {
    const s = d.pred.samples[state.sample];
    return { t0: d.pred.t0, dt: d.pred.dt, pos: s.pos, poss: s.poss };
  }
  return d[view];
}
const duration = (s) => (s.pos.length - 1) * s.dt;

function frameAt(s, t, out = new Float64Array(2 * N_AGENTS)) {
  const n = s.pos.length, f = Math.max(0, Math.min(t / s.dt, n - 1));
  const i = Math.floor(f), a = f - i;
  const A = s.pos[i], B = s.pos[Math.min(i + 1, n - 1)];
  for (let j = 0; j < 2 * N_AGENTS; j++) {
    const va = A[j], vb = B[j];
    out[j] = va == null ? (vb != null && a > 0.5 ? vb : NaN) : vb == null ? va : va + (vb - va) * a;
  }
  return out;
}

function average(s, t1, t2) {
  const i1 = Math.max(0, Math.floor(t1 / s.dt)), i2 = Math.min(s.pos.length - 1, Math.ceil(t2 / s.dt));
  const sum = new Float64Array(2 * N_AGENTS), cnt = new Float64Array(2 * N_AGENTS);
  for (let i = i1; i <= i2; i++) {
    const f = s.pos[i];
    for (let j = 0; j < 2 * N_AGENTS; j++) if (f[j] != null) { sum[j] += f[j]; cnt[j]++; }
  }
  return sum.map((v, j) => (cnt[j] ? v / cnt[j] : NaN));
}

function detectGoalkeepers() {
  const s = state.doc.half1 || state.doc.half2;
  const avg = average(s, 0, duration(s));
  state.h1Avg = state.doc.half1 ? average(state.doc.half1, 0, duration(state.doc.half1)) : null;
  if (state.doc.gk) { state.gk = state.doc.gk; return; } // video pipeline reserves the keeper slots
  state.gk = [0, 1].map((k) => {
    let best = k * N_TEAM, bestV = Infinity;
    for (let sl = k * N_TEAM; sl < (k + 1) * N_TEAM; sl++) {
      const v = dir(k) * avg[2 * sl];
      if (Number.isFinite(v) && v < bestV) { bestV = v; best = sl; }
    }
    return best;
  });
}

// ---------- metric time series for the chart ----------
function metricSeries(s, key) {
  const n = s.pos.length, out = [[], []];
  const xy = new Float64Array(2 * N_AGENTS);
  const win = Math.max(1, Math.round(300 / s.dt));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 2 * N_AGENTS; j++) xy[j] = s.pos[i][j] == null ? NaN : s.pos[i][j];
    if (key === "poss") {
      let a = 0, b = 0;
      for (let q = Math.max(0, i - win); q <= i; q++) { const p = s.poss[q]; if (p < N_TEAM) a++; else if (p < BALL) b++; }
      const v = a + b ? (100 * a) / (a + b) : NaN;
      out[0].push(v); out[1].push(Number.isFinite(v) ? 100 - v : NaN);
    } else {
      for (const k of [0, 1]) { const m = teamMetrics(xy, k); out[k].push(m ? m[key] : NaN); }
    }
  }
  const smooth = key === "poss" ? 1 : Math.max(1, Math.round(60 / s.dt));
  return out.map((arr) => rolling(arr, smooth));
}

function rolling(arr, w) {
  if (w <= 1) return arr;
  const out = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    let s = 0, c = 0;
    for (let q = Math.max(0, i - w + 1); q <= i; q++) if (Number.isFinite(arr[q])) { s += arr[q]; c++; }
    out[i] = c ? s / c : NaN;
  }
  return out;
}

function buildMetrics() {
  const d = state.doc, key = state.metric, m = {};
  if (d.half1) m.half1 = metricSeries(d.half1, key);
  if (d.half2) m.half2 = metricSeries(d.half2, key);
  if (d.pred) {
    const per = d.pred.samples.map((s) => metricSeries({ ...s, dt: d.pred.dt }, key));
    m.pred = [0, 1].map((k) => {
      const n = per[0][k].length, mean = [], sd = [];
      for (let i = 0; i < n; i++) {
        const v = per.map((p) => p[k][i]).filter(Number.isFinite);
        const mu = v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
        mean.push(mu);
        sd.push(v.length ? Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length) : NaN);
      }
      return { mean, sd };
    });
  }
  state.metrics = m;
}

// ---------- pitch ----------
const svg = $("#pitch");
const layers = {};
const svg2 = $("#pitch2"), layers2 = {}; // the what-if scenario, shown next to the actual 2nd half
const xyBuf2 = new Float64Array(2 * N_AGENTS);
const splitView = () => state.view === "half2" && !!state.doc.pred;

function drawPitch(target = svg, lay = layers) {
  target.innerHTML = "";
  const [L, W] = state.doc.pitch, hl = L / 2, hw = W / 2;
  const defs = el("defs", {}, target);
  for (const k of [0, 1]) {
    const mk = el("marker", { id: `arrow${k}`, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 4, markerHeight: 4, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: state.doc.teams[k].color }, mk);
  }
  const bg = el("g", {}, target);
  el("rect", { x: -58, y: -39, width: 116, height: 78, fill: "var(--pitch-a)" }, bg);
  const stripes = 12, sw = L / stripes;
  for (let i = 0; i < stripes; i += 2) el("rect", { x: -hl + i * sw, y: -hw, width: sw, height: W, fill: "var(--pitch-b)" }, bg);
  const g = el("g", { fill: "none", stroke: "var(--pitch-line)", "stroke-width": 0.25 }, target);
  el("rect", { x: -hl, y: -hw, width: L, height: W }, g);
  el("line", { x1: 0, y1: -hw, x2: 0, y2: hw }, g);
  el("circle", { cx: 0, cy: 0, r: 9.15 }, g);
  el("circle", { cx: 0, cy: 0, r: 0.35, fill: "var(--pitch-line)" }, g);
  for (const s of [-1, 1]) {
    const gl = s * hl;
    el("rect", { x: s < 0 ? gl : gl - 16.5, y: -20.16, width: 16.5, height: 40.32 }, g);
    el("rect", { x: s < 0 ? gl : gl - 5.5, y: -9.16, width: 5.5, height: 18.32 }, g);
    el("rect", { x: s < 0 ? gl - 2 : gl, y: -3.66, width: 2, height: 7.32 }, g);
    const spot = gl - s * 11, edge = gl - s * 16.5;
    el("circle", { cx: spot, cy: 0, r: 0.35, fill: "var(--pitch-line)" }, g);
    el("path", { d: s < 0 ? `M${edge},-7.31 A9.15,9.15 0 0 1 ${edge},7.31` : `M${edge},7.31 A9.15,9.15 0 0 1 ${edge},-7.31` }, g);
  }
  const arrows = el("g", { opacity: 0.55, "font-size": 2.2, fill: "var(--pitch-line)" }, target);
  const right = dir(0) > 0 ? 0 : 1; // the team attacking towards +x gets the left-hand caption
  el("text", { x: -hl + 2, y: hw + 3.2 }, arrows).textContent = `${state.doc.teams[right].name} attack →`;
  el("text", { x: hl - 2, y: hw + 3.2, "text-anchor": "end" }, arrows).textContent = `← ${state.doc.teams[1 - right].name} attack`;

  for (const name of ["avg", "hull", "links", "line", "trails", "fc", "players", "ball"]) lay[name] = el("g", {}, target);
  lay.dots = [];
  for (let a = 0; a < BALL; a++) {
    const k = a < N_TEAM ? 0 : 1, c = state.doc.teams[k].color;
    const grp = el("g", {}, lay.players);
    const ring = el("circle", { r: 2.1, fill: "none", stroke: "#ffe066", "stroke-width": 0.45, visibility: "hidden" }, grp);
    const dot = el("circle", { r: 1.25, fill: c, stroke: a === state.gk[k] ? "#fff" : "rgba(0,0,0,.55)", "stroke-width": a === state.gk[k] ? 0.4 : 0.25 }, grp);
    const lab = el("text", { "font-size": 1.25, "text-anchor": "middle", "dominant-baseline": "central", fill: "#0b0f0d", "font-weight": 700 }, grp);
    lab.textContent = a === state.gk[k] ? "GK" : String(a % N_TEAM);
    lay.dots.push({ grp, ring, dot, lab });
  }
  lay.ballDot = el("circle", { r: 0.75, fill: "#fff", stroke: "#000", "stroke-width": 0.2 }, lay.ball);
  // video only: where the ball was last seen while the detector misses it (in play, not out)
  lay.ballGhost = el("g", { visibility: "hidden" }, lay.ball);
  el("circle", { r: 1.2, fill: "none", stroke: "#fff", "stroke-width": 0.25, "stroke-dasharray": "0.5 0.4" }, lay.ballGhost);
  el("text", { "font-size": 1.7, "text-anchor": "middle", "dominant-baseline": "central", fill: "#fff", "font-weight": 700 }, lay.ballGhost).textContent = "?";
}

function renderPitch(xy, poss, lay = layers, withTrails = true) {
  const d = state.doc, Y = (y) => -y;
  for (let a = 0; a < BALL; a++) {
    const x = xy[2 * a], y = xy[2 * a + 1], o = lay.dots[a];
    const on = Number.isFinite(x);
    o.grp.setAttribute("visibility", on ? "visible" : "hidden");
    if (!on) { o.ring.setAttribute("visibility", "hidden"); continue; } // an explicit "visible" child would outlive its hidden group
    o.dot.setAttribute("cx", x); o.dot.setAttribute("cy", Y(y));
    o.lab.setAttribute("x", x); o.lab.setAttribute("y", Y(y));
    o.ring.setAttribute("cx", x); o.ring.setAttribute("cy", Y(y));
    o.ring.setAttribute("visibility", poss === a ? "visible" : "hidden");
  }
  const bx = xy[2 * BALL], by = xy[2 * BALL + 1];
  lay.ballDot.setAttribute("visibility", Number.isFinite(bx) ? "visible" : "hidden");
  if (Number.isFinite(bx)) { lay.ballDot.setAttribute("cx", bx); lay.ballDot.setAttribute("cy", Y(by)); }

  for (const n of ["hull", "links", "line", "trails"]) lay[n].innerHTML = "";
  for (const k of [0, 1]) {
    const c = d.teams[k].color, pts = outfield(xy, k);
    if (state.layers.hull && pts.length >= 3) {
      el("polygon", { points: hull(pts).map((p) => `${p[0]},${Y(p[1])}`).join(" "), fill: c, "fill-opacity": 0.12, stroke: c, "stroke-opacity": 0.7, "stroke-width": 0.3, "stroke-linejoin": "round" }, lay.hull);
    }
    if (state.layers.links) {
      for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
        const dd = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
        if (dd < LINK_M) el("line", { x1: pts[i][0], y1: Y(pts[i][1]), x2: pts[j][0], y2: Y(pts[j][1]), stroke: c, "stroke-width": 0.22, "stroke-opacity": 0.85 * (1 - dd / LINK_M) }, lay.links);
      }
    }
    if (state.layers.line) {
      const m = teamMetrics(xy, k);
      if (m) {
        const x = -dir(k) * halfLen() + dir(k) * m.defLine;
        el("line", { x1: x, y1: -d.pitch[1] / 2, x2: x, y2: d.pitch[1] / 2, stroke: c, "stroke-width": 0.35, "stroke-dasharray": "1.2 1", opacity: 0.9 }, lay.line);
      }
    }
  }
  if (state.layers.trails && withTrails) {
    const s = series(), steps = Math.max(2, Math.round(TRAIL_S / Math.min(s.dt, 1)));
    const buf = new Float64Array(2 * N_AGENTS), path = Array.from({ length: BALL }, () => []);
    for (let q = steps; q >= 0; q--) {
      const t = state.t - (q * TRAIL_S) / steps;
      if (t < 0) continue;
      frameAt(s, t, buf);
      for (let a = 0; a < BALL; a++) if (Number.isFinite(buf[2 * a])) path[a].push(`${buf[2 * a]},${Y(buf[2 * a + 1])}`);
    }
    path.forEach((p, a) => { if (p.length > 1) el("polyline", { points: p.join(" "), fill: "none", stroke: d.teams[a < N_TEAM ? 0 : 1].color, "stroke-width": 0.3, "stroke-opacity": 0.55 }, lay.trails); });
  }
}

function windowAverage() {
  const s = series(), w = state.windowMin * 60;
  return w ? average(s, Math.max(0, state.t - w), state.t) : average(s, 0, duration(s));
}

function renderAverages(avg) {
  layers.avg.innerHTML = "";
  if (!state.layers.formation && !state.layers.compare) return;
  const d = state.doc, Y = (y) => -y;
  for (const k of [0, 1]) {
    const c = d.teams[k].color;
    if (state.layers.compare && state.h1Avg && state.view !== "half1") {
      for (let a = k * N_TEAM; a < (k + 1) * N_TEAM; a++) {
        const x0 = state.h1Avg[2 * a], y0 = state.h1Avg[2 * a + 1], x1 = avg[2 * a], y1 = avg[2 * a + 1];
        if (!Number.isFinite(x0)) continue;
        el("circle", { cx: x0, cy: Y(y0), r: 1.1, fill: "none", stroke: c, "stroke-width": 0.25, "stroke-dasharray": "0.6 0.4", opacity: 0.8 }, layers.avg);
        if (Number.isFinite(x1) && Math.hypot(x1 - x0, y1 - y0) > 1) el("line", { x1: x0, y1: Y(y0), x2: x1, y2: Y(y1), stroke: c, "stroke-width": 0.3, "marker-end": `url(#arrow${k})`, opacity: 0.85 }, layers.avg);
      }
    }
    if (state.layers.formation) {
      const pts = outfield(avg, k);
      for (const p of pts) el("circle", { cx: p[0], cy: Y(p[1]), r: 1.6, fill: c, "fill-opacity": 0.28, stroke: c, "stroke-width": 0.3 }, layers.avg);
      const xs = pts.map((p) => ({ v: dir(k) * p[0], p })).sort((a, b) => a.v - b.v);
      const lines = formationLabel(avg, k).split("-").map(Number);
      let idx = 0;
      for (const n of lines) {
        if (!Number.isFinite(n)) break;
        const grp = xs.slice(idx, idx + n).map((o) => o.p).sort((a, b) => a[1] - b[1]);
        idx += n;
        if (grp.length > 1) el("polyline", { points: grp.map((p) => `${p[0]},${Y(p[1])}`).join(" "), fill: "none", stroke: c, "stroke-width": 0.35, "stroke-opacity": 0.75 }, layers.avg);
      }
    }
  }
}

// ---------- side panel ----------
const tracked = (xy, k) => { let n = 0; for (let a = k * N_TEAM; a < (k + 1) * N_TEAM; a++) if (Number.isFinite(xy[2 * a])) n++; return n; };
function renderPanel(xy, avg) {
  const d = state.doc, s = series();
  const i2 = Math.min(s.poss.length - 1, Math.floor(state.t / s.dt));
  let a = 0, b = 0;
  for (let i = 0; i <= i2; i++) { const p = s.poss[i]; if (p < N_TEAM) a++; else if (p < BALL) b++; }
  for (const k of [0, 1]) {
    const m = teamMetrics(xy, k), share = a + b ? (100 * (k === 0 ? a : b)) / (a + b) : NaN;
    const f = (v, u = "m", digits = 0) => (m && Number.isFinite(v) ? `${v.toFixed(digits)} ${u}` : "–");
    $(`#team${k}`).innerHTML = `
      <div class="head"><span class="sw" style="background:${d.teams[k].color}"></span>
        <span class="name">${d.teams[k].name}</span>
        <span class="formation" title="Formation inferred from average positions in the selected window">${formationLabel(avg, k)}</span></div>
      <dl class="stats">
        <dt>Width</dt><dd>${f(m && m.width)}</dd>
        <dt>Depth</dt><dd>${f(m && m.depth)}</dd>
        <dt>Shape area</dt><dd>${f(m && m.area, "m²")}</dd>
        <dt>Back line from own goal</dt><dd>${f(m && m.defLine)}</dd>
        <dt>Players tracked</dt><dd>${tracked(xy, k)} / 11</dd>
        <dt>Possession so far</dt><dd>${Number.isFinite(share) ? share.toFixed(0) + " %" : "–"}</dd>
      </dl>`;
  }
}

// ---------- chart ----------
const chart = $("#chart");
function chartContext() {
  const d = state.doc;
  if (state.view === "half1") return { real: state.metrics.half1, pred: null, dur: duration(d.half1), realDt: d.half1.dt, t0: 0 };
  const dur = Math.max(d.half2 ? duration(d.half2) : 0, d.pred ? duration({ pos: d.pred.samples[0].pos, dt: d.pred.dt }) : 0);
  return { real: state.metrics.half2, pred: state.metrics.pred, dur, realDt: d.half2 && d.half2.dt, predDt: d.pred && d.pred.dt, t0: 45 * 60 };
}

function renderChart() {
  const box = chart.getBoundingClientRect(), Wd = Math.max(200, box.width), Ht = Math.max(100, box.height);
  chart.setAttribute("viewBox", `0 0 ${Wd} ${Ht}`);
  chart.innerHTML = "";
  const ctx = chartContext(), d = state.doc, pad = { l: 38, r: 10, t: 8, b: 20 };
  const vals = [];
  const push = (arr) => arr && arr.forEach((v) => Number.isFinite(v) && vals.push(v));
  if (ctx.real) ctx.real.forEach(push);
  if (ctx.pred) ctx.pred.forEach((p) => p.mean.forEach((v, i) => Number.isFinite(v) && vals.push(v + p.sd[i], v - p.sd[i])));
  if (!vals.length) return;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const span = hi - lo || 1; lo -= span * 0.08; hi += span * 0.08;
  const X = (t) => pad.l + (t / ctx.dur) * (Wd - pad.l - pad.r), Yv = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (Ht - pad.t - pad.b);
  const axis = el("g", { fill: "var(--muted)", "font-size": 10 }, chart);
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    el("line", { x1: pad.l, x2: Wd - pad.r, y1: Yv(v), y2: Yv(v), stroke: "var(--border)", "stroke-width": 1 }, chart);
    el("text", { x: pad.l - 6, y: Yv(v) + 3, "text-anchor": "end" }, axis).textContent = v.toFixed(0);
  }
  const step = [5, 10, 30, 60, 300, 600].find((s) => ctx.dur / s <= 10) || 900;
  for (let t = 0; t <= ctx.dur + 1e-6; t += step) {
    el("text", { x: X(t), y: Ht - 5, "text-anchor": "middle" }, axis).textContent = step < 60 ? fmt(ctx.t0 + t) : `${Math.round((ctx.t0 + t) / 60)}'`;
  }
  const path = (arr, dt) => { let s = "", pen = false; arr.forEach((v, i) => { if (!Number.isFinite(v)) { pen = false; return; } s += `${pen ? "L" : "M"}${X(i * dt).toFixed(1)},${Yv(v).toFixed(1)}`; pen = true; }); return s; };
  for (const k of [0, 1]) {
    const c = d.teams[k].color;
    if (ctx.pred) {
      const p = ctx.pred[k], up = [], dn = [];
      p.mean.forEach((v, i) => { if (Number.isFinite(v)) { up.push(`${X(i * ctx.predDt)},${Yv(v + p.sd[i])}`); dn.unshift(`${X(i * ctx.predDt)},${Yv(v - p.sd[i])}`); } });
      if (up.length) el("polygon", { points: up.concat(dn).join(" "), fill: c, "fill-opacity": 0.16 }, chart);
      el("path", { d: path(p.mean, ctx.predDt), fill: "none", stroke: c, "stroke-width": 1.6, "stroke-dasharray": "5 4" }, chart);
    }
    if (ctx.real) el("path", { d: path(ctx.real[k], ctx.realDt), fill: "none", stroke: c, "stroke-width": 1.6 }, chart);
  }
  const cur = state.view === "half1" ? state.t : state.t;
  el("line", { x1: X(cur), x2: X(cur), y1: pad.t, y2: Ht - pad.b, stroke: "var(--text)", "stroke-width": 1, opacity: 0.6, id: "cursor" }, chart);
  chart.onclick = (e) => {
    const r = chart.getBoundingClientRect(), frac = (e.clientX - r.left - pad.l) / (r.width - pad.l - pad.r);
    seek(Math.max(0, Math.min(1, frac)) * ctx.dur);
  };
  const lg = $("#legend");
  lg.innerHTML = (ctx.real ? `<span><i></i>actual</span>` : "") + (ctx.pred ? `<span><i class="dash"></i>what-if mean ± 1σ (${d.pred.samples.length} scenarios)</span>` : "");
}

function moveCursor() {
  const c = chart.querySelector("#cursor");
  if (!c) return;
  const ctx = chartContext(), box = chart.getBoundingClientRect(), Wd = Math.max(200, box.width);
  const x = 38 + (Math.min(state.t, ctx.dur) / ctx.dur) * (Wd - 48);
  c.setAttribute("x1", x); c.setAttribute("x2", x);
}

// ---------- playback ----------
const fmt = (sec) => `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
const xyBuf = new Float64Array(2 * N_AGENTS);

function draw(force = false) {
  const s = series();
  state.t = Math.max(0, Math.min(state.t, duration(s)));
  frameAt(s, state.t, xyBuf);
  const poss = s.poss[Math.min(s.poss.length - 1, Math.round(state.t / s.dt))];
  renderPitch(xyBuf, poss);
  const ballMissed = renderBallGhost(s);
  renderForecast();
  if (splitView()) {
    const ps = series("pred");
    frameAt(ps, Math.min(state.t, duration(ps)), xyBuf2);
    renderPitch(xyBuf2, ps.poss[Math.min(ps.poss.length - 1, Math.round(state.t / ps.dt))], layers2, false);
    $("#badge2").textContent = `What-if scenario ${state.sample + 1}/${state.doc.pred.samples.length} · simulated from the 1st half`;
  }
  const now = performance.now();
  if (force || now - state.lastPanel > 200) {
    const avg = windowAverage();
    renderAverages(avg);
    renderPanel(xyBuf, avg);
    state.lastPanel = now;
  }
  $("#time").value = Math.round((1000 * state.t) / duration(s));
  $("#clock").textContent = fmt(s.t0 + state.t);
  moveCursor();
  syncVideo(force);
  const label = { half1: "1st half · actual tracking", half2: splitView() ? "2nd half · what actually happened" : "2nd half · actual tracking" }[state.view];
  $("#badge").textContent = ballMissed ? `${label} · ball not detected` : label;
}

// The video is half 1 only; it follows the timeline (native playback while playing, seeks while scrubbing).
function syncVideo(force) {
  const v = $("#video"), d = state.doc;
  if (!d.video || state.view !== "half1") { if (!v.paused) v.pause(); return; }
  const target = d.video.offset + state.t;
  if (state.playing && state.speed <= 16) {
    v.playbackRate = state.speed;
    if (v.paused) v.play().catch(() => {});
    if (Math.abs(v.currentTime - target) > 0.5 && !v.seeking) v.currentTime = target;
  } else {
    if (!v.paused) v.pause();
    if (force || Math.abs(v.currentTime - target) > 0.05) v.currentTime = target;
  }
}

function renderBallGhost(s) {
  const g = layers.ballGhost;
  if (!g) return false;
  const i = Math.min(s.pos.length - 1, Math.round(state.t / s.dt));
  if (!s.ball || s.ball[i] !== 1) { g.setAttribute("visibility", "hidden"); return false; }
  for (let q = i; q >= 0 && i - q < 600; q--) {
    const f = s.pos[q];
    if (f[2 * BALL] != null) {
      g.setAttribute("transform", `translate(${f[2 * BALL]},${-f[2 * BALL + 1]})`);
      g.setAttribute("visibility", "visible");
      return true;
    }
  }
  g.setAttribute("visibility", "hidden");
  return true;
}

let lastTs = null;
function tick(ts) {
  if (state.playing) {
    if (lastTs != null) state.t += ((ts - lastTs) / 1000) * state.speed;
    const f = state.doc.forecast;
    if (state.fc && state.fc.phase === "revealing" && state.t >= state.fc.t0 + f.steps * f.dt) finishReveal();
    else if (state.t >= duration(series())) { state.playing = false; $("#play").textContent = "▶"; }
    draw();
  }
  lastTs = ts;
  requestAnimationFrame(tick);
}

function seek(t) { clearForecast(); state.t = t; draw(true); }

// ---------- short-term forecast: pause -> predict -> reveal ----------
function fcData() {
  const f = state.doc.forecast;
  return f && f[state.view] ? f[state.view] : null;
}

function clearForecast() {
  if (!state.fc) return;
  if (state.fc.phase === "revealing") state.speed = state.fc.prevSpeed;
  state.fc = null;
  renderForecastCard();
}

function predict() {
  const rec = fcData();
  if (!rec) return;
  if (state.fc && state.fc.phase === "revealing") return;
  state.playing = false;
  $("#play").textContent = "▶";
  let lo = 0, hi = rec.t.length - 1, idx = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (rec.t[mid] <= state.t + 1e-6) { idx = mid; lo = mid + 1; } else hi = mid - 1; }
  state.fc = idx < 0 ? { phase: "early", idx: -1 } : { phase: "shown", idx, t0: rec.t[idx], key: state.view, mode: 0 };
  if (idx >= 0) state.t = rec.t[idx];
  renderForecastCard();
  draw(true);
}

function reveal() {
  if (!state.fc || state.fc.phase !== "shown") return;
  Object.assign(state.fc, { phase: "revealing", prevSpeed: state.speed });
  state.t = state.fc.t0;
  state.speed = 1;
  state.playing = true;
  $("#play").textContent = "❚❚";
  renderForecastCard();
  syncVideo(true);
}

function finishReveal() {
  const f = state.doc.forecast;
  state.playing = false;
  $("#play").textContent = "▶";
  state.speed = state.fc.prevSpeed;
  state.t = state.fc.t0 + f.steps * f.dt;
  state.fc.phase = "done";
  renderForecastCard();
  draw(true);
}

const rec_horizon = () => state.doc.forecast.steps * state.doc.forecast.dt;
const FUTURE = ["A", "B", "C"];
state.fcCircles = true;
const slotLabel = (a) => (a === state.gk[0] || a === state.gk[1] ? "GK" : String(a % N_TEAM));
const pct = (p) => `${Math.round(100 * p)}%`;
const circleRadii = (f, sel) => (f.radius90ByRank ? f.radius90ByRank[sel] : sel === 0 && f.radius90 ? f.radius90 : null);
// The self-adjusting radius of this moment for the selected future (it learns from misses already seen); exports
// without it fall back to the fixed per-rank radius.
const circleRadius = (f, rec, sel, idx) => (rec.radiusAci && rec.radiusAci[sel] ? rec.radiusAci[sel][0][idx] : (circleRadii(f, sel) || [null])[0]);

function renderForecast() {
  const g = layers.fc;
  if (!g) return;
  g.innerHTML = "";
  const fc = state.fc, rec = fcData();
  if (!fc || !rec || fc.idx < 0 || fc.key !== state.view) return;
  const d = state.doc, f = d.forecast, Y = (y) => -y, sel = fc.mode || 0;
  const colour = (a) => (a === BALL ? "#ffffff" : d.teams[a < N_TEAM ? 0 : 1].color);
  const start = frameAt(series(), fc.t0, new Float64Array(2 * N_AGENTS));
  const modes = rec.modes[fc.idx];
  const order = modes.map((_, k) => k).filter((k) => k !== sel).concat(sel); // selected future drawn last, on top
  for (const mi of order) {
    const top = mi === sel;
    for (let a = 0; a < N_AGENTS; a++) {
      const pts = Number.isFinite(start[2 * a]) ? [[start[2 * a], start[2 * a + 1]]] : [];
      for (const st of modes[mi]) if (st[2 * a] != null) pts.push([st[2 * a], st[2 * a + 1]]);
      if (pts.length < 2) continue;
      el("polyline", { points: pts.map((q) => `${q[0]},${Y(q[1])}`).join(" "), fill: "none", stroke: colour(a),
        "stroke-width": top ? 0.4 : 0.2, "stroke-opacity": top ? 0.95 : 0.3, "stroke-dasharray": top ? "0.9 0.5" : "0.4 0.4", "stroke-linecap": "round" }, g);
      if (top) {
        const e = pts[pts.length - 1];
        el("circle", { cx: e[0], cy: Y(e[1]), r: a === BALL ? 0.7 : 1.25, fill: "none", stroke: colour(a), "stroke-width": 0.3, "stroke-dasharray": "0.5 0.35" }, g);
      }
    }
  }
  const at4 = modes[0][0], radius = circleRadius(f, rec, sel, fc.idx);
  if (state.fcCircles && radius) {
    const st = modes[sel][0];
    for (let a = 0; a < BALL; a++) {
      if (st[2 * a] == null) continue;
      el("circle", { cx: st[2 * a], cy: Y(st[2 * a + 1]), r: radius, fill: colour(a), "fill-opacity": 0.05,
        stroke: colour(a), "stroke-opacity": 0.45, "stroke-width": 0.2 }, g);
    }
  }
  if (rec.who && fc.phase !== "done") { // players most likely on the ball at +4 s, ringed where the model expects them
    for (const [a, p] of rec.who[fc.idx][0].top) {
      if (at4[2 * a] == null) continue;
      el("circle", { cx: at4[2 * a], cy: Y(at4[2 * a + 1]), r: 2.3, fill: "none", stroke: "#ffe066", "stroke-width": 0.2 + 0.6 * p, opacity: 0.95 }, g);
      el("text", { x: at4[2 * a] + 2.7, y: Y(at4[2 * a + 1]) + 0.6, "font-size": 1.8, fill: "#ffe066", "font-weight": 700 }, g).textContent = pct(p);
    }
  }
  if (fc.phase === "revealing" || fc.phase === "done") { // solid = the path each player really ran so far
    const s = series(), buf = new Float64Array(2 * N_AGENTS), upto = Math.min(state.t, fc.t0 + rec_horizon());
    const path = Array.from({ length: N_AGENTS }, () => []);
    for (let t = fc.t0; t <= upto + 1e-6; t += 0.5) {
      frameAt(s, t, buf);
      for (let a = 0; a < N_AGENTS; a++) if (Number.isFinite(buf[2 * a])) path[a].push(`${buf[2 * a]},${Y(buf[2 * a + 1])}`);
    }
    path.forEach((pts, a) => {
      if (pts.length > 1) el("polyline", { points: pts.join(" "), fill: "none", stroke: colour(a), "stroke-width": 0.45, "stroke-opacity": 0.95, "stroke-linecap": "round" }, g);
    });
  }
  if (fc.phase === "done") { // whiskers from each predicted end point to where the player really was
    const end = modes[sel][modes[sel].length - 1], now = frameAt(series(), state.t, new Float64Array(2 * N_AGENTS));
    for (let a = 0; a < N_AGENTS; a++) {
      if (end[2 * a] == null || !Number.isFinite(now[2 * a])) continue;
      el("line", { x1: end[2 * a], y1: Y(end[2 * a + 1]), x2: now[2 * a], y2: Y(now[2 * a + 1]), stroke: "#ffe066", "stroke-width": 0.25, opacity: 0.9 }, g);
    }
  }
}

// ---------- results: model vs simple guesses on every match with forecasts ----------
function combineHalves(fs) {
  const hs = Object.values(fs.halves);
  const avg = (get, weight) => {
    let s = 0, w = 0;
    for (const h of hs) { const v = get(h), ww = weight(h); if (v != null && Number.isFinite(v)) { s += v * ww; w += ww; } }
    return w ? s / w : NaN;
  };
  const n = (h) => h.n, held = (h) => h.held_n[0];
  return {
    model: avg((h) => h.model[0], n), still: avg((h) => h.still[0], n), cv: avg((h) => h.cv[0], n),
    team: avg((h) => h.decisions.team[0], held), teamBase: avg((h) => h.decisions.team_base[0], held),
    top3: avg((h) => h.decisions.top3[0], held), top3Base: avg((h) => h.decisions.top3_base[0], held),
    cover: avg((h) => h.coverage90[0], n), moments: hs.reduce((a, h) => a + h.n, 0), heldN: hs.reduce((a, h) => a + h.held_n[0], 0),
    coverA: avg((h) => (h.coverage90Adaptive ? h.coverage90Adaptive[0] : NaN), n),
    radiusA: avg((h) => (h.radiusAciMean ? h.radiusAciMean[0] : NaN), n),
  };
}

// Pool matches: position error and circle coverage weighted by moments, decisions by moments with a ball holder.
function poolMatches(rows) {
  const w = (get, weight) => {
    let s = 0, ww = 0;
    for (const r of rows) { const v = get(r), x = weight(r); if (Number.isFinite(v)) { s += v * x; ww += x; } }
    return ww ? s / ww : NaN;
  };
  const mo = (r) => r.moments, he = (r) => r.heldN;
  return {
    model: w((r) => r.model, mo), still: w((r) => r.still, mo), cv: w((r) => r.cv, mo),
    team: w((r) => r.team, he), teamBase: w((r) => r.teamBase, he), top3: w((r) => r.top3, he), top3Base: w((r) => r.top3Base, he),
    cover: w((r) => r.cover, mo), coverA: w((r) => r.coverA, mo), radiusA: w((r) => r.radiusA, mo), r: w((r) => r.r, mo),
    moments: rows.reduce((a, r) => a + r.moments, 0),
  };
}

const matchTitle = (name) => (name.startsWith("Sample_Game") ? `Metrica ${name.replace(/_/g, " ")}` : name.replace(/^(pff|sc|dfl)_\S+ /, ""));
// how the current checkpoint used a match (exports before roles existed only know held-out vs independent)
const roleOf = (fs) => fs.role || (fs.independent ? "independent" : "val");
const ROLE_TAG = { train: "seen in training", val: "held out from training", independent: "independent test" };

function renderResults() {
  const data = (state.index || []).filter((e) => e.forecastSummary).map((e) => ({
    name: matchTitle(e.name), role: roleOf(e.forecastSummary), indep: roleOf(e.forecastSummary) === "independent",
    r: e.forecastSummary.radius90, ...combineHalves(e.forecastSummary),
  })).filter((d) => d.role !== "train").sort((a, b) => a.indep - b.indep || a.name.localeCompare(b.name));
  if (!data.length) { $("#resultsBody").innerHTML = `<p class="fc-note">No match has forecasts yet: run h2predict.forecast_ui.</p>`; return; }
  // pool the held-out matches (they picked the checkpoint) and, if several, the independent ones
  const held = data.filter((d) => !d.indep), indep = data.filter((d) => d.indep);
  const groups = (held.length ? [{ ...poolMatches(held), name: `Held-out matches · ${held.length}`, indep: false }] : [])
    .concat(indep.length > 1 ? [{ ...poolMatches(indep), name: `Independent tests · ${indep.length} matches`, indep: true }] : indep);
  const bar = (v, max, cls, label) => `<div class="rbar ${cls}"><span style="width:${Math.max(1.5, (100 * v) / max).toFixed(1)}%"></span><em>${label}</em></div>`;
  const tag = (indep) => `<small class="tag${indep ? " indep" : ""}">${ROLE_TAG[indep ? "independent" : "val"]}</small>`;
  const head = (g) => `<div class="rmatch">${g.name}${tag(g.indep)}</div>`;
  const maxErr = Math.max(...groups.flatMap((g) => [g.model, g.still, g.cv]));
  const block = (title, sub, body) => `<section class="rblock"><h4>${title}</h4><p class="rsub">${sub}</p>${body}</section>`;
  const circleBars = (g) => `<div class="rbar b"><span style="width:${(100 * g.cover).toFixed(1)}%"></span><i style="left:90%"></i><em>Fixed circle, r = ${g.r.toFixed(1)} m: ${pct(g.cover)} inside</em></div>`
    + (Number.isFinite(g.coverA) ? `<div class="rbar m"><span style="width:${(100 * g.coverA).toFixed(1)}%"></span><i style="left:90%"></i><em>Self-adjusting, mean r = ${g.radiusA.toFixed(1)} m: ${pct(g.coverA)} inside</em></div>` : "");
  const f1 = (v) => (Number.isFinite(v) ? v.toFixed(1) : "–"), p0 = (v) => (Number.isFinite(v) ? pct(v) : "–");
  const table = `<section class="rblock"><h4>Per match</h4><p class="rsub">Bold = model; then the simple guesses. Both halves, every 4 s.</p>
    <div class="rtable-wrap"><table class="rtable"><thead><tr><th>Match</th><th>Position error at +4 s<small>model · keep velocity · stand still</small></th>
      <th>Team on the ball<small>model · nearest team</small></th><th>Holder in top 3<small>model · 3 nearest</small></th>
      <th>90% circles<small>fixed · self-adjusting</small></th></tr></thead><tbody>
    ${data.map((d) => `<tr><td>${d.name}${d.indep ? tag(true) : ""}</td><td><b>${f1(d.model)}</b> · ${f1(d.cv)} · ${f1(d.still)} m</td>
      <td><b>${p0(d.team)}</b> · ${p0(d.teamBase)}</td><td><b>${p0(d.top3)}</b> · ${p0(d.top3Base)}</td><td>${p0(d.cover)} · <b>${p0(d.coverA)}</b></td></tr>`).join("")}
    </tbody></table></div></section>`;
  $("#resultsBody").innerHTML =
    block("Where every player will be in 4 s", "Average distance from the predicted to the real position. Lower is better.",
      groups.map((g) => head(g) + bar(g.model, maxErr, "m", `Model ${g.model.toFixed(1)} m`) + bar(g.cv, maxErr, "b", `Keep velocity ${g.cv.toFixed(1)} m`) + bar(g.still, maxErr, "b", `Stand still ${g.still.toFixed(1)} m`)).join("")) +
    block("Which team will have the ball in 4 s", "Moments when someone had the ball 4 s later. Higher is better.",
      groups.map((g) => head(g) + bar(g.team, 1, "m", `Model ${pct(g.team)}`) + bar(g.teamBase, 1, "b", `Team nearest the ball now ${pct(g.teamBase)}`)).join("")) +
    block("Who will have the ball in 4 s", "The real holder is among the model's 3 most likely players. Higher is better.",
      groups.map((g) => head(g) + bar(g.top3, 1, "m", `Model's top 3 ${pct(g.top3)}`) + bar(g.top3Base, 1, "b", `3 players nearest the ball ${pct(g.top3Base)}`)).join("")) +
    block("Do the 90% circles keep their promise?", "Share of players inside the circle around their predicted position (most likely future). Target: 90% (white line). Self-adjusting circles widen or tighten every 4 s from the misses already seen.",
      groups.map((g) => head(g) + circleBars(g)).join("")) +
    table +
    `<p class="fc-note">Held-out matches were never trained on but did choose the checkpoint. Independent tests were used for nothing (the Metrica match also comes from another provider and league). There the model still beats both simple guesses; fixed circles are too tight because that provider's tracking differs, and the self-adjusting circles widen to keep the 90% promise. The guarantee is an average over a match's moments, not for every single moment.</p>`;
}

function decisionHtml(rec, i, f) {
  if (!rec.who) return "";
  const d = state.doc, [now, later] = rec.who[i];
  const team = (k, p) => `<span style="width:${(100 * p).toFixed(1)}%;background:${d.teams[k].color}">${p >= 0.18 ? `${d.teams[k].name} ${pct(p)}` : ""}</span>`;
  const who = now.top.map(([a, p]) => `<b style="color:${d.teams[a < N_TEAM ? 0 : 1].color}">${d.teams[a < N_TEAM ? 0 : 1].name} #${slotLabel(a)}</b> ${pct(p)}`).join(" · ");
  const control = now.held != null
    ? `<div class="fc-control"><span>Chance someone controls the ball in ${f.dt} s</span><b>${pct(now.held)}</b>${later && later.held != null ? `<small>in ${2 * f.dt} s: ${pct(later.held)}</small>` : ""}</div>`
    : "";
  return `<div class="fc-decision">${control}
    <div class="fc-label">If someone has the ball in ${f.dt} s, it is</div>
    <div class="fc-bar">${team(0, now.team[0])}${team(1, now.team[1])}</div>
    <div class="fc-who">Most likely on it: ${who}</div>
    ${later ? `<div class="fc-sub">In ${2 * f.dt} s: ${d.teams[0].name} ${pct(later.team[0])} · ${d.teams[1].name} ${pct(later.team[1])}</div>` : ""}
  </div>`;
}

function renderForecastCard() {
  const card = $("#fcCard"), rec = fcData(), f = state.doc.forecast;
  card.hidden = !rec;
  if (!rec) return;
  const fc = state.fc, hz = f.steps * f.dt, sum = f.summary[state.view], i = fc && fc.idx >= 0 ? fc.idx : -1, sel = (fc && fc.mode) || 0;
  $("#fcPredict").textContent = `Predict next ${hz} s`;
  $("#fcReveal").disabled = !(fc && fc.phase === "shown");
  const help = {
    none: `Pause anywhere and press Predict. The model sees the last ${f.history_s} s of tracking and draws where every player and the ball will be in ${f.dt} s and ${hz} s. Then press Reveal or ▶ to compare with what really happened.`,
    early: `The model needs ${f.history_s} s of history: move a little later in the half.`,
    shown: "Football branches, so the model gives a menu of futures with honest odds. Dashed = the selected future, faint = the others, yellow rings = who is likely on the ball.",
    revealing: "Playing what actually happened: solid lines are the real runs, dashed lines the prediction.",
    done: "Yellow lines join each predicted end point to where that player really was.",
  };
  $("#fcHelp").textContent = help[fc ? fc.phase : "none"];
  let html = "";
  if (i >= 0) {
    html += `<div class="fc-futures">${rec.prob[i].map((p, k) => `<button class="chip${k === sel ? " on" : ""}" data-mode="${k}">Future ${FUTURE[k]} · ${pct(p)}</button>`).join("")}</div>`;
    const radius = circleRadius(f, rec, sel, i), adaptive = !!(rec.radiusAci && rec.radiusAci[sel]);
    if (f.radius90) {
      html += `<label class="check small"><input type="checkbox" id="fcCircles"${state.fcCircles ? " checked" : ""}${radius ? "" : " disabled"}>
        90% circles at +${f.dt} s${radius ? ` (r = ${radius.toFixed(1)} m${adaptive ? ", self-adjusting" : ""})` : " (future A only)"}</label>`;
    }
    html += decisionHtml(rec, i, f);
  }
  if (fc && fc.phase === "done") {
    const e = rec.err, mErr = e.modes ? e.modes[i][sel] : e.model[i];
    const row = (name, vals, cls = "") => `<tr class="${cls}"><td>${name}</td>${vals.map((v) => `<td>${v == null ? "–" : v.toFixed(1) + " m"}</td>`).join("")}</tr>`;
    html += `<table class="fc-table"><thead><tr><th>Avg error, this moment</th>${mErr.map((_, s) => `<th>+${(s + 1) * f.dt} s</th>`).join("")}</tr></thead>
      <tbody>${row(`Future ${FUTURE[sel]}`, mErr, "model")}${row("Stand still", e.still[i])}${row("Keep velocity", e.cv[i])}</tbody></table>`;
  }
  const notes = [`Average error at +${f.dt} s over this half (${rec.t.length} moments): <b>model ${sum.model[0].toFixed(1)} m</b> · stand still ${sum.still[0].toFixed(1)} m · keep velocity ${sum.cv[0].toFixed(1)} m.`];
  if (sum.heldAuc) {
    notes.push(`"Controls the ball" = a player within 1.5 m of a slow ball, true in ${pct(sum.heldRate[0])} of moments here (the rest: passes, loose and dead balls). It is a rough probability, not a yes/no: it ranks controlled vs not-controlled moments correctly ${pct(sum.heldAuc[0])} of the time (50% = guessing) and is roughly calibrated on the World Cup matches (says 45% → happens 48%)${f.independent ? ", but on this match it under-states 20–40% chances" : ""}.`);
  }
  if (sum.decisions) {
    const dd = sum.decisions;
    notes.push(`When someone held the ball ${f.dt} s later: team called right <b>${pct(dd.team[0])}</b> vs ${pct(dd.team_base[0])} for "team nearest the ball now"; the real holder was in the top 3 <b>${pct(dd.top3[0])}</b> vs ${pct(dd.top3_base[0])} for "3 nearest players".`);
  }
  if (sum.coverage90AdaptiveByRank && rec.radiusAci && rec.radiusAci[sel] && i >= 0) {
    const fixedR = circleRadii(f, sel)[0], fixedCov = sum.coverage90ByRank ? sum.coverage90ByRank[sel][0] : sum.coverage90[0];
    notes.push(`Future ${FUTURE[sel]}'s 90% circles start at ${fixedR.toFixed(1)} m (sized on ${f.calibGames} other matches) and adjust every ${f.dt} s from the misses already seen: <b>${rec.radiusAci[sel][0][i].toFixed(1)} m</b> at this moment. On this match: fixed circle ${pct(fixedCov)} inside, self-adjusting <b>${pct(sum.coverage90AdaptiveByRank[sel][0])}</b>.`);
  } else if (f.radius90 && sum.coverage90) {
    const cov = sum.coverage90ByRank ? sum.coverage90ByRank[sel][0] : sum.coverage90[0];
    notes.push(`90% circles were sized on ${f.calibGames} other matches; on this match they contained the real player <b>${pct(cov)}</b> of the time${cov < 0.85 ? " — too tight here: this data comes from another provider" : ""}.`);
  }
  if (f.independent) notes.push("Independent test: another data provider and league, used neither for training nor for choosing the model.");
  else if (f.heldOut) notes.push("Held out from training (used only to choose the checkpoint).");
  html += notes.map((n) => `<p class="fc-note">${n}</p>`).join("");
  $("#fcResult").innerHTML = html;
  card.querySelectorAll("[data-mode]").forEach((b) => (b.onclick = () => { state.fc.mode = +b.dataset.mode; renderForecastCard(); draw(true); }));
  const cb = $("#fcCircles");
  if (cb) cb.onchange = () => { state.fcCircles = cb.checked; draw(true); };
}

function setView(v) {
  if (state.fc && state.fc.phase === "revealing") { state.playing = false; $("#play").textContent = "▶"; }
  state.fc = null;
  // After a match switch the previous view may not exist in the new doc: start that view from 0.
  const prevClock = state.doc[state.view] ? series().t0 + state.t : null;
  state.view = v;
  document.querySelectorAll("#view button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  $("#sampleWrap").style.display = v === "half2" && state.doc.pred ? "" : "none";
  $("#media").classList.toggle("has-video", !!state.doc.video && v === "half1");
  $("#media").classList.toggle("split", v === "half2" && !!state.doc.pred);
  const s = series();
  state.t = prevClock == null ? 0 : Math.max(0, Math.min(prevClock - s.t0, duration(s)));
  renderChart();
  renderForecastCard();
  draw(true);
}

// ---------- boot ----------
async function loadMatch(file) {
  state.playing = false;
  $("#play").textContent = "▶";
  if (state.fc && state.fc.phase === "revealing") state.speed = state.fc.prevSpeed;
  state.fc = null;
  state.doc = await (await fetch(`data/${file}`, { cache: "no-cache" })).json(); // revalidate: exports change
  const video = $("#video");
  if (state.doc.video) video.src = `data/${state.doc.video.file}`; else video.removeAttribute("src");
  video.onloadedmetadata = () => syncVideo(true);
  detectGoalkeepers();
  drawPitch();
  drawPitch(svg2, layers2);
  buildMetrics();
  const d = state.doc;
  document.querySelectorAll("#view button").forEach((b) => { b.disabled = !d[b.dataset.view]; });
  $("#sample").innerHTML = d.pred ? d.pred.samples.map((_, i) => `<option value="${i}">${i + 1}</option>`).join("") : "";
  state.sample = 0; state.t = 0;
  setView(d.half1 ? "half1" : "half2");
}

async function boot() {
  let index = [];
  try { index = await (await fetch("data/index.json", { cache: "no-store" })).json(); } catch (e) { /* no exports yet */ }
  state.index = index;
  if (!index.length) {
    document.querySelector(".layout").innerHTML = `<div class="empty">No matches exported yet. Run <code>python -m h2predict.export_ui</code> to add one to <code>ui/data/</code>.</div>`;
    return;
  }
  $("#match").innerHTML = index.map((e) => `<option value="${e.file}">${e.name}${e.hasForecast ? " · forecasts" : ""}${e.hasVideo ? " · video" : ""}</option>`).join("");
  $("#match").onchange = (e) => loadMatch(e.target.value);
  document.querySelectorAll("#view button").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
  $("#sample").onchange = (e) => { state.sample = +e.target.value; draw(true); };
  $("#play").onclick = togglePlay;
  $("#time").oninput = (e) => seek((+e.target.value / 1000) * duration(series()));
  $("#speed").onchange = (e) => (state.speed = +e.target.value);
  $("#metric").onchange = (e) => { state.metric = e.target.value; buildMetrics(); renderChart(); };
  $("#window").onchange = (e) => { state.windowMin = +e.target.value; draw(true); };
  document.querySelectorAll("[data-layer]").forEach((c) => (c.onchange = () => { state.layers[c.dataset.layer] = c.checked; draw(true); }));
  $("#fcPredict").onclick = predict;
  $("#resultsBtn").onclick = () => { renderResults(); $("#results").showModal(); };
  $("#resultsClose").onclick = () => $("#results").close();
  $("#fcReveal").onclick = reveal;
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    else if (e.code === "KeyP") predict();
    else if (e.code === "KeyR") reveal();
  });
  window.addEventListener("resize", () => renderChart());
  // open on a World Cup match with forecasts (real team names, current model), then any match with forecasts
  const first = index.find((e) => e.hasForecast && !(e.forecastSummary && e.forecastSummary.independent))
    || index.find((e) => e.hasForecast) || index[0];
  $("#match").value = first.file;
  await loadMatch(first.file);
  requestAnimationFrame(tick);
}

function togglePlay() {
  // With a forecast on screen, play means "show me what really happened" and keeps the dashed prediction.
  if (state.fc && state.fc.phase === "shown") { reveal(); return; }
  if (state.fc && state.fc.phase === "revealing") {
    state.playing = !state.playing;
    $("#play").textContent = state.playing ? "❚❚" : "▶";
    syncVideo(true);
    return;
  }
  clearForecast();
  if (state.t >= duration(series())) state.t = 0;
  state.playing = !state.playing;
  $("#play").textContent = state.playing ? "❚❚" : "▶";
  syncVideo(true);
}

boot();
