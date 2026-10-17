"use strict";
// Better runs: for the moments h2predict.runs ranked highest, draw the real run of one attacker, the run the
// value model prefers, and where the other players are predicted to be 8 s later in either case.

const $ = (s) => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const N_TEAM = 11, BALL = 22, PITCH = [106, 68];
const state = { doc: null, sel: 0, show: "best" };

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
const Y = (y) => -y;
const ok = (p) => p && p[0] != null && p[1] != null;
const clock = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const pct = (v) => `${Math.round(100 * v)}%`;
const label = (a) => `${state.doc.teams[a < N_TEAM ? 0 : 1]} #${a % N_TEAM}`;
const colour = (a) => (a === BALL ? "#ffffff" : state.doc.colors[a < N_TEAM ? 0 : 1]);

function drawPitch(g) {
  const [L, W] = PITCH, line = { fill: "none", stroke: "rgba(255,255,255,0.7)", "stroke-width": 0.3 };
  el("rect", { x: -L / 2 - 2, y: -W / 2 - 2, width: L + 4, height: W + 4, fill: "#1f5132" }, g);
  el("rect", { x: -L / 2, y: -W / 2, width: L, height: W, ...line }, g);
  el("line", { x1: 0, y1: -W / 2, x2: 0, y2: W / 2, ...line }, g);
  el("circle", { cx: 0, cy: 0, r: 9.15, ...line }, g);
  for (const s of [-1, 1]) {
    el("rect", { x: s > 0 ? L / 2 - 16.5 : -L / 2, y: -20.16, width: 16.5, height: 40.32, ...line }, g);
    el("rect", { x: s > 0 ? L / 2 - 5.5 : -L / 2, y: -9.16, width: 5.5, height: 18.32, ...line }, g);
  }
}

function arrow(g, pts, attrs) {
  el("polyline", { points: pts.map((p) => `${p[0]},${Y(p[1])}`).join(" "), fill: "none", "stroke-linecap": "round",
    "stroke-linejoin": "round", "marker-end": "url(#head)", ...attrs }, g);
}

function draw() {
  const svg = $("#pitch");
  svg.innerHTML = "";
  const defs = el("defs", {}, svg);
  const mk = el("marker", { id: "head", viewBox: "0 0 10 10", refX: 7, refY: 5, markerWidth: 4, markerHeight: 4, orient: "auto-start-reverse" }, defs);
  el("path", { d: "M1 1L8 5L1 9", fill: "none", stroke: "context-stroke", "stroke-width": 2 }, mk);
  drawPitch(el("g", {}, svg));
  const tp = state.doc && state.doc.turningPoints[state.sel];
  if (!tp) return;
  const a = tp.agent, start = tp.start, react = tp[state.show].others;
  const g = el("g", {}, svg);
  for (let i = 0; i < BALL; i++) {  // predicted reaction: where everyone else is expected 8 s later
    if (i === a || !ok(start[i]) || !ok(react[i])) continue;
    el("line", { x1: start[i][0], y1: Y(start[i][1]), x2: react[i][0], y2: Y(react[i][1]), stroke: colour(i),
      "stroke-width": 0.25, "stroke-dasharray": "0.6 0.5", opacity: 0.8 }, g);
    el("circle", { cx: react[i][0], cy: Y(react[i][1]), r: 1.1, fill: "none", stroke: colour(i), "stroke-width": 0.3 }, g);
  }
  for (let i = 0; i <= BALL; i++) {  // everyone now
    if (!ok(start[i])) continue;
    el("circle", { cx: start[i][0], cy: Y(start[i][1]), r: i === BALL ? 0.7 : 1.3, fill: colour(i),
      stroke: i === a ? "#ffe066" : "rgba(0,0,0,0.5)", "stroke-width": i === a ? 0.6 : 0.2 }, g);
    if (i < BALL) el("text", { x: start[i][0], y: Y(start[i][1]) + 0.6, "font-size": 1.6, "text-anchor": "middle",
      fill: "#0e1412", "font-weight": 700 }, g).textContent = String(i % N_TEAM);
  }
  const real = [start[a], ...tp.realFuture.map((s) => s[a])].filter(ok);
  arrow(g, real, { stroke: "#ffffff", "stroke-width": 0.55 });
  arrow(g, [start[a], ...tp.best.run], { stroke: "#8fd16a", "stroke-width": 0.65, "stroke-dasharray": "1.4 0.8" });
  if (state.doc.value === "irl") {
    const costly = tp.epvDelta > 0 ? `the better run would also have raised the team's EPV (${tp.epvDelta >= 0 ? "+" : ""}${tp.epvDelta.toFixed(3)}): worth a look` : `the real run kept a higher team EPV (${tp.epvDelta.toFixed(3)}): possibly a deliberate, creative choice`;
    $("#caption").innerHTML = `<b>${label(a)}</b> at ${clock(tp.clock)}, ${tp.group}. White = the real run (8 s); dashed green = the same-length run that best fits what players in this role usually aim for (learned from 24 other matches). ` +
      `Hollow circles = where the others are forecast 8 s later (the same for both runs). Gap to that objective ${tp.gain.toFixed(2)}, mostly: ${tp.why || "several small things"}. ` +
      `By EPV, ${costly}. The real run beat ${pct(tp.realBeats)} of the ${tp.nAlts} alternatives.`;
    return;
  }
  $("#caption").innerHTML = `<b>${label(a)}</b> at ${clock(tp.clock)}. White = the real run (8 s); dashed green = the run the value model prefers. ` +
    `Hollow circles = where the others are predicted 8 s later if ${label(a)} makes the ${state.show === "best" ? "suggested" : "real"} run. ` +
    `${state.doc.valueLabel || "Value"}: ${tp.ctReal.toFixed(3)} → ${tp.ctBest.toFixed(3)} (+${tp.gain.toFixed(3)}); the suggestion beats the real run in ${pct(tp.robust)} of the predicted futures; ` +
    `the real run was better than ${pct(tp.realBeats)} of the ${tp.nAlts} alternatives tried.`;
}

function renderList() {
  const d = state.doc;
  const s = d.summary;
  if (d.value === "irl") {
    $("#summary").innerHTML = `${s.playerMoments} runs of both teams, in and out of possession, scored with the objective learned from 24 other matches (inverse RL). ` +
      `On this held-out match the real runs beat <b>${pct(s.realRunBeatsAlternatives)}</b> of the same-length alternatives (chance: 50%). Largest gaps between a run and the player's own group objective:`;
    $("#list").innerHTML = d.turningPoints.map((tp, k) => `<button class="tp${k === state.sel ? " on" : ""}" data-k="${k}"><b>${clock(tp.clock)}</b> · ${label(tp.agent)}
      <small>${tp.group} · ${tp.why || ""} · EPV ${tp.epvDelta >= 0 ? "+" : ""}${tp.epvDelta.toFixed(3)}</small></button>`).join("");
    $("#advice").innerHTML = (d.advice || []).slice(0, 8).map((x) => `<li><b>${label(x.agent)}</b>: clearly below their group's objective in ${pct(x.better)} of ${x.moments} runs, mostly ${x.why || "mixed reasons"}; the better run also had a higher team EPV in ${pct(x.costlyShare)} of those.</li>`).join("");
    document.querySelectorAll("[data-k]").forEach((b) => (b.onclick = () => { state.sel = +b.dataset.k; renderList(); draw(); }));
    return;
  }
  const where = s.realRunBeatsWhereItMatters != null ? ` (<b>${pct(s.realRunBeatsWhereItMatters)}</b> in the quarter of runs where the direction matters most)` : "";
  $("#summary").innerHTML = `${s.moments} attacking moments, ${s.playerMoments} player runs scored. The real runs beat ` +
    `<b>${pct(s.realRunBeatsAlternatives)}</b> of the same-length alternatives on average${where}; a clearly better run (in at least 60% of the predicted futures) ` +
    `exists for ${pct(s.robustBetterShare)} of the runs. ` +
    (s.realRunBeatsAlternatives < 0.55 ? `<b>Research prototype:</b> the real runs score close to chance against plausible alternatives, so these suggestions are questions to review, not advice. ` : "") +
    `Moments where a different run would have gained the most:`;
  $("#list").innerHTML = d.turningPoints.map((tp, k) => {
    return `<button class="tp${k === state.sel ? " on" : ""}" data-k="${k}"><b>${clock(tp.clock)}</b> · ${label(tp.agent)}
      <small>value +${tp.gain.toFixed(3)} · better in ${pct(tp.robust)} of futures</small></button>`;
  }).join("") || `<p class="summary">No moment passed the robustness filter.</p>`;
  const dir = (v, pos, neg) => (Math.abs(v) < 1 ? "" : `${Math.abs(v).toFixed(0)} m ${v > 0 ? pos : neg}`);
  $("#advice").innerHTML = (d.advice || []).slice(0, 6).map((x) => {
    const how = [dir(x.forward, "further forward", "deeper"), dir(x.inside, "more central", "wider")].filter(Boolean).join(", ") || "a different angle";
    return `<li><b>${label(x.agent)}</b>: a clearly better run in ${pct(x.better)} of ${x.moments} attacking moments, typically ${how}.</li>`;
  }).join("");
  document.querySelectorAll("[data-k]").forEach((b) => (b.onclick = () => { state.sel = +b.dataset.k; renderList(); draw(); }));
}

async function load(file) {
  state.doc = await (await fetch(`data/runs/${file}`, { cache: "no-cache" })).json();
  state.sel = 0;
  renderList();
  draw();
}

async function boot() {
  $("#legend").innerHTML = `<span><i class="swatch" style="border-color:#fff"></i>real run</span>
    <span><i class="swatch" style="border-color:#8fd16a;border-top-style:dashed"></i>suggested run</span>
    <span>○ others 8 s later (predicted)</span>`;
  let index = [];
  try { index = await (await fetch("data/runs/index.json", { cache: "no-cache" })).json(); } catch { index = []; }
  if (!index.length) { $("#summary").textContent = "No results yet: run python -m h2predict.runs --match pff:3834"; draw(); return; }
  $("#match").innerHTML = index.map((e) => `<option value="${e.file}">${e.match.replace(/^(pff|sc|dfl)_\S+ /, "")}</option>`).join("");
  $("#match").onchange = (e) => load(e.target.value);
  document.querySelectorAll("#react button").forEach((b) => (b.onclick = () => {
    state.show = b.dataset.show;
    document.querySelectorAll("#react button").forEach((x) => x.classList.toggle("on", x === b));
    draw();
  }));
  const first = index.find((e) => e.file.endsWith("_irl.json")) || index[0];  // the learned objective first
  $("#match").value = first.file;
  await load(first.file);
}

boot();
