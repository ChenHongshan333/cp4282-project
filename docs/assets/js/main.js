// The interactive parts of the project page: tables from data/tables.json, the single-blob demo, the
// budget bars, the training curves, the player of fitted models and the arc demo.

import { WIDTH, HEIGHT, SPECS, numbersPerGaussian, parseModel, sliceAt, renderCPU, centreAt, arcCentre, arcFrame,
         ARC } from "./gauss.js";
import { makeRenderer } from "./gl.js";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const COLOUR = { M0: "#52514e", M1: "#2a78d6", M2: "#eb6834", M3: "#1baf7a", D3: "#6250d6", D7: "#e34948" };
const NAME = { M0: "M0 straight", M1: "M1 curved", M2: "M2 shrinking", M3: "M3 both", D3: "D3 degree 3", D7: "D7 degree 7" };
const MINUS = "−";
const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : x.toFixed(d).replace("-", MINUS));
const signed = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : (x >= 0 ? "+" : MINUS) + Math.abs(x).toFixed(d));

// "no-cache" revalidates with the server, so a re-exported file is never read from a stale cache.
const json = (url) => fetch(url, { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); });
const DATA = { tables: json("data/tables.json"), models: json("data/models.json"), runs: json("data/runs.json") };

// Segmented buttons: <div class="seg"> with <button data-v>; calls onChange(value).
function segment(el, values, current, onChange, label = (v) => v) {
  if (values) el.innerHTML = values.map((v) => `<button data-v="${v}">${label(v)}</button>`).join("");
  const set = (v) => $$("button", el).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === String(v))));
  set(current);
  el.onclick = (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled) return;
    set(b.dataset.v);
    onChange(b.dataset.v);
  };
  return set;
}

// ===================================================================================== tables
async function fillTables() {
  const tables = await DATA.tables;
  for (const el of $$("[data-table]")) {
    const t = tables[el.dataset.table];
    if (!t) continue;
    el.classList.add("table-wrap");
    // data-shade: grey out gains within ±0.3 dB, the margin of the pre-registered test
    const shade = "shade" in el.dataset;
    const cell = (c) => {
      const text = String(c).replace(/✓/g, '<span class="pass">✓</span>').replace(/✗/g, '<span class="fail">✗</span>')
        .replace(/^passes$/, '<span class="pass">passes</span>').replace(/^fails$/, '<span class="fail">fails</span>');
      const classes = [];
      if (/^[\d+−\-—]/.test(String(c))) classes.push("num");
      if (shade && Math.abs(parseFloat(String(c).replace(MINUS, "-"))) <= 0.3) classes.push("small");
      return `<td${classes.length ? ` class="${classes.join(" ")}"` : ""}>${text}</td>`;
    };
    el.innerHTML = `<table><caption>${el.dataset.caption || ""}</caption><thead><tr>${t.head
      .map((h, i) => `<th${i > 0 && /^[\d+−\-—]/.test(String(t.rows[0]?.[i] ?? "")) ? ' class="num"' : ""}>${h}</th>`)
      .join("")}</tr></thead><tbody>${t.rows.map((r) => `<tr>${r.map(cell).join("")}</tr>`).join("")}</tbody></table>`;
  }
  // A long caption inside a table makes the table as wide as the caption. Move every caption above
  // its table, so that a table is only as wide as its columns.
  for (const caption of $$("table > caption")) {
    const div = document.createElement("div");
    div.className = "tcap";
    div.innerHTML = caption.innerHTML;
    caption.closest("table").before(div);
    caption.remove();
  }
  // the tables change the height of the page: go to the linked section again
  if (location.hash) $(location.hash)?.scrollIntoView();
}

// ===================================================================================== one blob
function blobWidget() {
  const canvas = $("#blob-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const image = new Float32Array(WIDTH * HEIGHT * 3);
  const ids = ["t", "v", "a", "smin"];
  const el = Object.fromEntries(ids.map((k) => [k, $(`#blob-${k}`)]));
  const curve = $("#blob-curve"), shrink = $("#blob-shrink");
  const sigmaT = 1.6, scale = 26;

  function draw() {
    const t = +el.t.value, v = +el.v.value, a = curve.checked ? +el.a.value : 0, smin = +el.smin.value;
    el.a.disabled = !curve.checked;
    el.smin.disabled = !shrink.checked;
    $("#blob-t-out").textContent = fmt(t, 2) + " frames";
    $("#blob-v-out").textContent = v + " px/frame";
    $("#blob-a-out").textContent = curve.checked ? fmt(a, 2) + " px/frame²" : "off";
    $("#blob-smin-out").textContent = shrink.checked ? fmt(smin, 2) : "off";
    const centre = (tau) => [WIDTH / 2 + v * tau * 0.9, HEIGHT / 2 + a * tau * tau];
    const w = Math.exp(-0.5 * (t / sigmaT) ** 2);
    const s = shrink.checked ? Math.max(w, smin) : 1;
    const r = Math.sqrt(s);
    const [x, y] = centre(t);
    // one Gaussian, slanted, drawn by the same renderer as the models
    // the weight of the blob at each pixel (times its visibility w), by the renderer of the models
    const slice = { count: 1, mu: new Float32Array([x, y]), chol: new Float32Array([scale * 1.3 * r, scale * 0.5 * r, scale * 0.7 * r]),
                    colour: new Float32Array([w, w, w]) };
    renderCPU(slice, image);
    const out = ctx.createImageData(WIDTH, HEIGHT);
    const bg = [0.985, 0.985, 0.98], blob = [0.92, 0.42, 0.12];
    for (let p = 0, q = 0; p < image.length; p += 3, q += 4) {
      const alpha = Math.min(image[p], 1);
      for (let c = 0; c < 3; c++) out.data[q + c] = Math.round(255 * (bg[c] * (1 - alpha) + blob[c] * alpha));
      out.data[q + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    // the path over the clip, and the centre at the 10 frames of a clip
    ctx.strokeStyle = "rgba(20,20,20,0.35)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let k = 0; k <= 90; k++) { const [px, py] = centre(-4.5 + k / 10); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke();
    ctx.setLineDash([]);
    for (let k = -4.5; k <= 4.5; k += 1) {
      const [px, py] = centre(k);
      ctx.fillStyle = `rgba(20,20,20,${0.25 + 0.6 * Math.exp(-0.5 * (k / sigmaT) ** 2)})`;
      ctx.beginPath(); ctx.arc(px, py, 2.5, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.fillStyle = "#141414";
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 2 * Math.PI); ctx.fill();
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText("peak time t₀", WIDTH / 2 - 30, HEIGHT / 2 + 54);
    $("#blob-w").textContent = fmt(w, 2);
    $("#blob-s").textContent = shrink.checked ? `max(w, s_min) = ${fmt(s, 2)}` : "1 (no shrinking)";
    $("#blob-cost").textContent = curve.checked ? "14 (12 + acceleration)" : "12";
  }
  [...Object.values(el), curve, shrink].forEach((e) => e.addEventListener("input", draw));
  draw();
}

// ===================================================================================== budget
function budgetWidget() {
  const range = $("#budget-range");
  if (!range) return;
  const budgets = [3000, 6000, 12000, 24000, 96000];
  const models = ["M0", "M1", "M2", "M3", "D3", "D7"];
  const extra = { M0: "", M1: "+ acceleration", M2: "+ shrinking (free)", M3: "+ both", D3: "degree-3 path", D7: "degree-7 path (VeGaS default)" };
  const bars = $("#budget-bars");
  bars.innerHTML = models.map((m) => `<span><span class="chip" style="background:${COLOUR[m]}"></span>${m} <span style="color:var(--muted)">${numbersPerGaussian(m)} numbers ${extra[m]}</span></span><div><div class="bar" data-m="${m}" style="background:${COLOUR[m]}"></div></div><span class="n" data-n="${m}"></span>`).join("");
  const draw = () => {
    const b = budgets[+range.value];
    $("#budget-out").textContent = b.toLocaleString("en") + " numbers = " + (4 * b / 1000).toLocaleString("en") + " kB";
    const max = Math.floor(b / 12);
    for (const m of models) {
      const n = Math.floor(b / numbersPerGaussian(m));
      $(`[data-m="${m}"]`, bars).style.width = `${(100 * n) / max}%`;
      $(`[data-n="${m}"]`, bars).textContent = n.toLocaleString("en") + " blobs";
    }
  };
  range.addEventListener("input", draw);
  draw();
}

// ===================================================================================== training curves
async function trainingWidget() {
  const root = $("#w-train");
  if (!root) return;
  const runs = await DATA.runs;
  const long = runs.filter((r) => r.stage === "long_test");
  const short = runs.filter((r) => r.stage === "test" && r.stride === 3 && r.frames === 10 && r.start === 0);
  const models = ["M0", "M1", "M2", "M3"];
  let xmode = "step", cursor = 2;               // cursor: index into the trace (without step 0)
  $("#train-legend").innerHTML = models.map((m) => `<span class="key"><span class="line" style="background:${COLOUR[m]}"></span>${NAME[m]}</span>`).join("")
    + `<span class="key"><svg width="12" height="12"><path d="M6 1 L11 6 L6 11 L1 6Z" fill="#888"/></svg>separate 3,000-step fit</span>`;
  segment($("#train-x"), null, "step", (v) => { xmode = v; draw(); });
  $("#train-seq").onchange = draw;
  $("#train-budget").onchange = draw;

  function series() {
    const seq = $("#train-seq").value, budget = +$("#train-budget").value;
    const out = {};
    for (const m of models) {
      const rs = long.filter((r) => r.sequence === seq && r.budget === budget && r.model === m).sort((a, b) => a.seed - b.seed);
      const n = rs[0].trace.length;
      const pts = [];
      for (let i = 1; i < n; i++) {
        pts.push({ step: rs[0].trace[i][0] + 1, time: rs.reduce((s, r) => s + r.trace[i][2], 0) / rs.length / 60,
                   psnr: rs.reduce((s, r) => s + r.trace[i][1], 0) / rs.length, seeds: Object.fromEntries(rs.map((r) => [r.seed, r.trace[i][1]])) });
      }
      const sh = short.filter((r) => r.sequence === seq && r.budget === budget && r.model === m);
      out[m] = { pts, short: { psnr: sh.reduce((s, r) => s + r.psnr, 0) / sh.length, time: sh.reduce((s, r) => s + r.train_seconds, 0) / sh.length / 60 } };
    }
    return out;
  }

  const W = 600, H = 340, M = { l: 48, r: 14, t: 12, b: 40 };
  function draw() {
    const s = series();
    const xs = (p) => (xmode === "step" ? p.step / 1000 : p.time);
    const all = models.flatMap((m) => s[m].pts);
    const xmax = Math.max(...all.map(xs));
    const ys = all.map((p) => p.psnr).concat(models.map((m) => s[m].short.psnr));
    const ymin = Math.floor(Math.min(...ys) - 0.3), ymax = Math.ceil(Math.max(...ys) + 0.2);
    const X = (v) => M.l + ((W - M.l - M.r) * v) / xmax, Y = (v) => H - M.b - ((H - M.t - M.b) * (v - ymin)) / (ymax - ymin);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="PSNR during training">`;
    const yt = [];
    for (let v = ymin; v <= ymax; v += ymax - ymin > 6 ? 2 : 1) yt.push(v);
    svg += yt.map((v) => `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#ebeae6"/><text x="${M.l - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`).join("");
    const step = xmode === "step" ? 5 : xmax > 6 ? 2 : 1;
    for (let v = 0; v <= xmax + 1e-9; v += step) svg += `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="#8f8d86"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${v}</text>`;
    svg += `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="#8f8d86"/>`;
    svg += `<text x="${(W + M.l) / 2}" y="${H - 6}" text-anchor="middle">${xmode === "step" ? "training step (thousands)" : "training time (minutes, laptop RTX 4060)"}</text>`;
    svg += `<text transform="translate(12 ${(H - M.b) / 2}) rotate(-90)" text-anchor="middle">PSNR on fitted frames (dB)</text>`;
    for (const m of models) {
      svg += `<path d="${s[m].pts.map((p, i) => `${i ? "L" : "M"}${X(xs(p)).toFixed(1)} ${Y(p.psnr).toFixed(1)}`).join("")}" fill="none" stroke="${COLOUR[m]}" stroke-width="2"/>`;
      const sx = X(xmode === "step" ? 3 : s[m].short.time), sy = Y(s[m].short.psnr);
      svg += `<path d="M${sx} ${sy - 6} L${sx + 6} ${sy} L${sx} ${sy + 6} L${sx - 6} ${sy}Z" fill="${COLOUR[m]}" stroke="#fff" stroke-width="1.5"/>`;
    }
    const cx = X(xs(s.M0.pts[cursor]));
    svg += `<line x1="${cx}" x2="${cx}" y1="${M.t}" y2="${H - M.b}" stroke="#141414" stroke-dasharray="3 3"/>`;
    svg += `<rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" fill="transparent" class="hit" style="cursor:ew-resize"/>`;
    svg += "</svg>";
    const chart = $("#train-chart");
    chart.innerHTML = svg;
    const hit = $(".hit", chart);
    const move = (e) => {
      const rect = chart.querySelector("svg").getBoundingClientRect();
      const v = ((e.clientX - rect.left) / rect.width * W - M.l) / (W - M.l - M.r) * xmax;
      let best = 0;
      s.M0.pts.forEach((p, i) => { if (Math.abs(xs(p) - v) < Math.abs(xs(s.M0.pts[best]) - v)) best = i; });
      if (best !== cursor) { cursor = best; draw(); }
    };
    hit.addEventListener("pointerdown", (e) => { hit.setPointerCapture(e.pointerId); move(e); });
    hit.addEventListener("pointermove", (e) => { if (e.buttons || e.pointerType === "mouse") move(e); });
    drawBars(s);
  }

  function drawBars(s) {
    const p0 = s.M0.pts[cursor];
    const gains = models.slice(1).map((m) => {
      const d = Object.keys(p0.seeds).map((k) => s[m].pts[cursor].seeds[k] - p0.seeds[k]);
      return { m, mean: d.reduce((a, b) => a + b, 0) / d.length, min: Math.min(...d), max: Math.max(...d) };
    });
    const W2 = 380, H2 = 340, L = 96, R = 48, T = 40, B = 40;
    const lim = Math.max(1, ...gains.map((g) => Math.max(Math.abs(g.min), Math.abs(g.max)))) * 1.1;
    const X = (v) => L + ((W2 - L - R) * (v + lim)) / (2 * lim);
    const rowH = (H2 - T - B) / 3;
    let svg = `<svg viewBox="0 0 ${W2} ${H2}" role="img" aria-label="Gain over M0 at the cursor">`;
    svg += `<text x="${L}" y="16" font-weight="600" fill="#141414">Gain over M0 at ${(p0.step / 1000).toFixed(0)}k steps (${p0.time.toFixed(1)} min)</text>`;
    svg += `<rect x="${X(-0.3)}" y="${T - 6}" width="${X(0.3) - X(-0.3)}" height="${H2 - T - B + 12}" fill="#f1f0ec"/>`;
    for (let v = -Math.floor(lim); v <= lim; v += 1) svg += `<line x1="${X(v)}" x2="${X(v)}" y1="${T - 6}" y2="${H2 - B + 6}" stroke="${v === 0 ? "#52514e" : "#ebeae6"}"/><text x="${X(v)}" y="${H2 - B + 22}" text-anchor="middle">${signed(v, 0).replace("+0", "0").replace(MINUS + "0", "0")}</text>`;
    svg += `<text x="${(W2 + L) / 2}" y="${H2 - 4}" text-anchor="middle">dB (grey band: ±0.3 dB)</text>`;
    gains.forEach((g, i) => {
      const y = T + rowH * i + rowH / 2;
      const x0 = X(Math.min(0, g.mean)), x1 = X(Math.max(0, g.mean));
      svg += `<text x="${L - 8}" y="${y + 4}" text-anchor="end" fill="#141414">${NAME[g.m]}</text>`;
      svg += `<rect x="${x0}" y="${y - 11}" width="${Math.max(1, x1 - x0)}" height="22" rx="3" fill="${COLOUR[g.m]}"/>`;
      svg += `<line x1="${X(g.min)}" x2="${X(g.max)}" y1="${y}" y2="${y}" stroke="#141414" stroke-width="1.2"/><line x1="${X(g.min)}" x2="${X(g.min)}" y1="${y - 5}" y2="${y + 5}" stroke="#141414"/><line x1="${X(g.max)}" x2="${X(g.max)}" y1="${y - 5}" y2="${y + 5}" stroke="#141414"/>`;
      const lx = g.mean >= 0 ? Math.max(x1, X(g.max)) + 6 : Math.min(x0, X(g.min)) - 6;
      svg += `<text x="${lx}" y="${y - 14}" text-anchor="${g.mean >= 0 ? "start" : "end"}" fill="#141414" font-weight="600">${signed(g.mean)}</text>`;
    });
    $("#train-bars").innerHTML = svg + "</svg>";
  }
  draw();
}

// ===================================================================================== models, frames
const modelCache = new Map();
async function loadModel(id) {
  if (!modelCache.has(id)) {
    modelCache.set(id, (async () => {
      const { order, models } = await DATA.models;
      const info = models[id];
      const buffer = await fetch("data/" + info.file, { cache: "no-cache" }).then((r) => r.arrayBuffer());
      return Object.assign(parseModel(buffer, info, order), { info, id });
    })());
  }
  return modelCache.get(id);
}
const imageCache = new Map();
function loadImage(url) {
  if (!imageCache.has(url)) {
    imageCache.set(url, new Promise((resolve, reject) => { const img = new Image(); img.onload = () => resolve(img); img.onerror = reject; img.src = url; }));
  }
  return imageCache.get(url);
}
const pad4 = (k) => String(k).padStart(4, "0");

let renderer = null;
const getRenderer = () => (renderer ||= makeRenderer());

// Draw a model at time t into a 2D canvas (480 x 270), optionally as the error against a target image.
function drawModel(ctx, model, t, mode = "image", target = null) {
  const r = getRenderer();
  r.draw(sliceAt(model, t));
  if (mode === "error" && target) r.setTarget(target);
  r.present(mode === "error" && target ? "error" : "image");
  ctx.drawImage(r.canvas, 0, 0);
  return r;
}

// Outline of a binary mask image, drawn on an overlay.
async function maskOutline(url) {
  const img = await loadImage(url);
  const c = document.createElement("canvas");
  c.width = WIDTH; c.height = HEIGHT;
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0, WIDTH, HEIGHT);
  const src = g.getImageData(0, 0, WIDTH, HEIGHT).data;
  const out = g.createImageData(WIDTH, HEIGHT);
  const on = (x, y) => x >= 0 && y >= 0 && x < WIDTH && y < HEIGHT && src[4 * (y * WIDTH + x)] > 127;
  let x0 = WIDTH, y0 = HEIGHT, x1 = 0, y1 = 0;
  for (let y = 0; y < HEIGHT; y++) for (let x = 0; x < WIDTH; x++) {
    if (!on(x, y)) continue;
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    if (on(x - 1, y) && on(x + 1, y) && on(x, y - 1) && on(x, y + 1)) continue;
    const q = 4 * (y * WIDTH + x);
    out.data[q] = 255; out.data[q + 1] = 230; out.data[q + 2] = 0; out.data[q + 3] = 255;
  }
  g.putImageData(out, 0, 0);
  return { canvas: c, box: x1 >= x0 ? [x0, y0, x1, y1] : null };
}

// ===================================================================================== player
async function playerWidget() {
  const root = $("#w-player");
  if (!root) return;
  const { models } = await DATA.models;
  // clips: key "Sequence_kStart" -> available (stride, budget, iterations, model)
  const clips = {};
  for (const [id, m] of Object.entries(models)) {
    if (m.sequence.startsWith("arc")) continue;
    const key = `${m.sequence}_k${m.start}`;
    (clips[key] ||= []).push({ id, ...m });
  }
  const clipLabel = { Jockey_k0: "Jockey (horse)", ShakeNDry_k0: "ShakeNDry (dog)", YachtRide_k0: "YachtRide (boat)", Bosphorus_k0: "Bosphorus (small boat)",
                      ShakeNDry_k30: "ShakeNDry from 30 (confirmatory)", Jockey_k120: "Jockey from 120 (confirmatory)", Jockey_k200: "Jockey from 200 (confirmatory)" };
  const order = Object.keys(clipLabel).filter((k) => clips[k]);
  $("#pl-clip").innerHTML = order.map((k) => `<option value="${k}">${clipLabel[k]}</option>`).join("");
  const state = { clip: "ShakeNDry_k0", stride: 3, budget: 24000, iterations: 30000, sides: ["M0", "M1"], t: 4 + 1 / 3,
                  mode: "image", mask: $("#pl-mask").checked, paths: $("#pl-paths").checked, box: null, playing: false };
  $("#pl-clip").value = state.clip;
  $("#pl-t").value = state.t;
  const panels = $$("#pl-panels .panel");
  const lens = $$("#pl-lens canvas").map((c) => c.getContext("2d"));
  const ctxs = panels.map((p) => $("canvas", p).getContext("2d"));
  const ovs = panels.map((p) => $("canvas.ov", p).getContext("2d"));
  let token = 0;

  const available = () => clips[state.clip];
  function options() {
    const av = available();
    const uniq = (f) => [...new Set(av.map(f))].sort((a, b) => a - b);
    const strides = uniq((m) => m.stride);
    if (!strides.includes(state.stride)) state.stride = strides[0];
    const budgets = uniq((m) => m.budget);
    if (!budgets.includes(state.budget)) state.budget = budgets[budgets.length - 1];
    const iters = uniq((m) => m.iterations);
    if (!iters.includes(state.iterations)) state.iterations = Math.max(...iters);
    segment($("#pl-stride"), strides, state.stride, (v) => { state.stride = +v; refresh(); });
    segment($("#pl-budget"), budgets, state.budget, (v) => { state.budget = +v; refresh(); }, (v) => (+v).toLocaleString("en"));
    segment($("#pl-iters"), iters, state.iterations, (v) => { state.iterations = +v; refresh(); }, (v) => (+v === 30000 ? "fully trained" : "3,000 steps"));
    const ms = av.filter((m) => m.stride === state.stride && m.budget === state.budget && m.iterations === state.iterations).map((m) => m.model);
    const all = ["M0", "M1", "M2", "M3", "D3", "D7"].filter((m) => ms.includes(m));
    state.sides = state.sides.map((s, i) => (all.includes(s) ? s : all[Math.min(i, all.length - 1)]));
    $$("#pl-panels select").forEach((sel, i) => {
      sel.innerHTML = all.map((m) => `<option value="${m}">${NAME[m]}</option>`).join("");
      sel.value = state.sides[i];
      sel.onchange = () => { state.sides[i] = sel.value; render(); };
    });
    // the time slider moves between real frames of the sequence: steps of 1 / stride in clip time
    const t = $("#pl-t");
    t.step = 1 / state.stride;
    state.t = Math.round(state.t * state.stride) / state.stride;
    t.value = state.t;
    let ticks = "";
    for (let k = 0; k <= 9 * state.stride; k++) {
      const pos = (100 * k) / (9 * state.stride), fit = k % state.stride === 0;
      ticks += `<i class="${fit ? "fit" : ""}" style="left:${pos}%"></i>`;
      if (fit) ticks += `<span style="left:${pos}%;top:9px">${k / state.stride}</span>`;
    }
    $("#pl-ticks").innerHTML = ticks;
  }
  const modelId = (name) => available().find((m) => m.model === name && m.stride === state.stride && m.budget === state.budget && m.iterations === state.iterations)?.id;

  function refresh() { options(); state.box = null; render(); }

  async function render() {
    const my = ++token;
    const [seq, k] = state.clip.split("_k");
    const start = +k;
    const frame = start + Math.round(state.t * state.stride);
    const fitted = Math.abs(state.t - Math.round(state.t)) < 1e-6;
    $("#pl-frame-no").textContent = frame;
    $("#pl-kind").innerHTML = fitted ? '<span class="badge fit">fitted</span>' : '<span class="badge between">between</span>';
    $("#pl-t-out").textContent = `t = ${state.t.toFixed(2)}`;
    $("#pl-status").textContent = "Loading…";
    let target, ms, mask;
    try {
      [target, ...ms] = await Promise.all([loadImage(`data/frames/${state.clip}/${pad4(frame)}.webp`), ...state.sides.map((s) => loadModel(modelId(s)))]);
      mask = await maskOutline(`data/masks/${state.clip}/${pad4(frame)}.png`).catch(() => null);
    } catch (e) {
      $("#pl-status").textContent = "Could not load: " + e.message;
      return;
    }
    if (my !== token) return;
    const r = getRenderer();
    ms.forEach((m, i) => drawModel(ctxs[i], m, state.t, state.mode, target));
    if (state.mode === "error") {
      // the error of the real frame against itself is 0: show it black
      ctxs[2].fillStyle = "#000"; ctxs[2].fillRect(0, 0, WIDTH, HEIGHT);
    } else ctxs[2].drawImage(target, 0, 0, WIDTH, HEIGHT);
    // PSNR from the evaluation of the fit (whole frame) at this time
    ms.forEach((m, i) => {
      const e = m.info.times.find((x) => Math.abs(x.t - state.t) < 1e-3);
      $(`[data-psnr="${i}"]`, root).textContent = !e ? "not evaluated"
        : `frame ${e.psnr.toFixed(2)} · object ${e.psnr_in == null ? "—" : e.psnr_in.toFixed(2)} dB`;
    });
    if (!state.box && mask?.box) {
      const [x0, y0, x1, y1] = mask.box;
      const w = Math.min(160, Math.max(112, (x1 - x0) * 0.75)), h = (w * 9) / 16;
      const cx = (x0 + x1) / 2, cy = seq === "Jockey" ? y0 + 0.72 * (y1 - y0) : (y0 + y1) / 2;
      state.box = [Math.max(0, Math.min(WIDTH - w, cx - w / 2)), Math.max(0, Math.min(HEIGHT - h, cy - h / 2)), w, h];
    }
    state.box ||= [180, 100, 128, 72];
    // overlays: mask outline, paths, zoom box
    ovs.forEach((o, i) => {
      o.clearRect(0, 0, WIDTH, HEIGHT);
      if (state.mask && mask) o.drawImage(mask.canvas, 0, 0);
      if (state.paths && i < 2) drawPaths(o, ms[i], state.t, [0, 0, WIDTH, HEIGHT], 1);
      const [x, y, w, h] = state.box;
      o.strokeStyle = "#fff"; o.lineWidth = 2.5; o.strokeRect(x, y, w, h);
      o.strokeStyle = "#141414"; o.lineWidth = 1; o.strokeRect(x, y, w, h);
    });
    lens.forEach((l, i) => {
      const [x, y, w, h] = state.box;
      l.imageSmoothingEnabled = false;
      l.drawImage(ctxs[i].canvas, x, y, w, h, 0, 0, WIDTH, HEIGHT);
      if (state.mask && mask) l.drawImage(mask.canvas, x, y, w, h, 0, 0, WIDTH, HEIGHT);
      if (state.paths && i < 2) drawPaths(l, ms[i], state.t, state.box, 2);
    });
    $("#pl-status").textContent = `${ms.map((m) => `${m.info.model}: ${m.n.toLocaleString("en")} blobs`).join(" · ")} · seed 0 · `
      + `PSNR of the whole frame and of the object, from the evaluation of each fit · renderer: ${r.precision === "cpu" ? "CPU" : "WebGL"}`;
  }

  // Paths of the visible blobs whose centre is inside the zoom box now, from 1.5 frames before to
  // 1.5 frames after (clip time), drawn into the view [vx, vy, vw, vh] of a 480 x 270 canvas.
  function drawPaths(o, model, t, [vx, vy, vw, vh], width) {
    const s = sliceAt(model, t);
    const [bx, by, bw, bh] = state.box;
    const sx = WIDTH / vw, sy = HEIGHT / vh, V = (x, y) => [(x - vx) * sx, (y - vy) * sy];
    const a = Math.max(0, t - 1.5), b = Math.min(9, t + 1.5);
    o.lineWidth = width;
    for (let j = 0; j < s.count; j++) {
      const i = s.index[j], w = Math.exp(-0.5 * ((t - model.t0[i]) / model.sigma_t[i]) ** 2);
      const [cx, cy] = [s.mu[2 * j], s.mu[2 * j + 1]];
      if (w < 0.3 || cx < bx || cy < by || cx > bx + bw || cy > by + bh) continue;
      o.strokeStyle = `rgba(255,255,255,${0.35 + 0.6 * w})`;
      o.beginPath();
      for (let q = 0; q <= 30; q++) { const [x, y] = V(...centreAt(model, i, a + ((b - a) * q) / 30)); q ? o.lineTo(x, y) : o.moveTo(x, y); }
      o.stroke();
      const [x, y] = V(cx, cy);
      o.fillStyle = "#ffd200";
      o.fillRect(x - width - 0.5, y - width - 0.5, 2 * width + 1, 2 * width + 1);
    }
  }

  // controls
  $("#pl-clip").onchange = () => { state.clip = $("#pl-clip").value; refresh(); };
  $("#pl-t").oninput = () => { state.t = +$("#pl-t").value; render(); };
  segment($("#pl-mode"), null, "image", (v) => { state.mode = v; render(); });
  $("#pl-mask").onchange = () => { state.mask = $("#pl-mask").checked; render(); };
  $("#pl-paths").onchange = () => { state.paths = $("#pl-paths").checked; render(); };
  panels.forEach((p) => {
    const stack = $(".stack", p);
    const place = (e) => {
      const rect = stack.getBoundingClientRect();
      const [, , w, h] = state.box;
      const x = ((e.clientX - rect.left) / rect.width) * WIDTH, y = ((e.clientY - rect.top) / rect.height) * HEIGHT;
      state.box = [Math.max(0, Math.min(WIDTH - w, x - w / 2)), Math.max(0, Math.min(HEIGHT - h, y - h / 2)), w, h];
      render();
    };
    stack.style.cursor = "crosshair";
    stack.addEventListener("pointerdown", (e) => { stack.setPointerCapture(e.pointerId); place(e); });
    stack.addEventListener("pointermove", (e) => { if (e.buttons) place(e); });
  });
  let timer = null;
  $("#pl-play").onclick = () => {
    if (timer) { clearInterval(timer); timer = null; $("#pl-play").textContent = "Play"; return; }
    $("#pl-play").textContent = "Pause";
    timer = setInterval(() => {
      state.t = state.t + 1 / state.stride > 9 + 1e-6 ? 0 : Math.round((state.t + 1 / state.stride) * state.stride) / state.stride;
      $("#pl-t").value = state.t;
      render();
    }, 160);
  };
  options();
  render();
}

// ===================================================================================== arc
async function arcWidget() {
  const root = $("#w-arc");
  if (!root) return;
  const angles = [0, 90, 180, 270, 360];
  const state = { angle: angles[+$("#arc-angle").value], budget: 24000, t: +$("#arc-t").value,
                  paths: $("#arc-paths").checked, zoom: $("#arc-zoom").checked };
  const panels = $$("#arc-panels .panel");
  const ctxs = panels.map((p) => $("canvas", p).getContext("2d"));
  const ovs = panels.map((p) => $("canvas.ov", p).getContext("2d"));
  const off = [0, 1, 2].map(() => { const c = document.createElement("canvas"); c.width = WIDTH; c.height = HEIGHT; return c.getContext("2d"); });
  const truthCanvas = off[2];
  let token = 0;
  let ticks = "";
  for (let k = 0; k <= 27; k++) {
    const pos = (100 * k) / 27, fit = k % 3 === 0;
    ticks += `<i class="${fit ? "fit" : ""}" style="left:${pos}%"></i>`;
    if (fit) ticks += `<span style="left:${pos}%;top:9px">${k / 3}</span>`;
  }
  $("#arc-ticks").innerHTML = ticks;

  // the view: the whole frame, or a box around the path of the ball
  function view() {
    if (!state.zoom) return [0, 0, WIDTH, HEIGHT];
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let f = 0; f <= 27; f += 0.25) { const [x, y] = arcCentre(state.angle, f); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const w = Math.max(x1 - x0 + 40, ((y1 - y0 + 40) * 16) / 9), h = (w * 9) / 16;
    return [cx - w / 2, cy - h / 2, w, h];
  }

  async function render() {
    const my = ++token;
    $("#arc-angle-out").textContent = `${state.angle}°`;
    $("#arc-t-out").textContent = `t = ${state.t.toFixed(2)}`;
    const fitted = Math.abs(state.t - Math.round(state.t)) < 1e-6;
    $("#arc-kind").innerHTML = fitted ? '<span class="badge fit">fitted</span>' : '<span class="badge between">between</span>';
    const ids = ["M0", "M1"].map((m) => `arcs${state.angle}_k0_s3_${m}_B${state.budget}_seed0_30k`);
    let ms;
    try { ms = await Promise.all(ids.map(loadModel)); } catch (e) { $("#arc-status").textContent = "Could not load: " + e.message; return; }
    if (my !== token) return;
    const truth = arcFrame(state.angle, state.t);
    const [bx, by] = arcCentre(state.angle, ARC.stride * state.t);
    const ball = new Uint8Array(WIDTH * HEIGHT);
    for (let y = 0; y < HEIGHT; y++) for (let x = 0; x < WIDTH; x++) ball[y * WIDTH + x] = Math.hypot(x + 0.5 - bx, y + 0.5 - by) < ARC.radius + 1 ? 1 : 0;
    const r = getRenderer();
    const psnrs = ms.map((m, i) => {
      r.draw(sliceAt(m, state.t));
      r.present("image");
      off[i].drawImage(r.canvas, 0, 0);
      const img = r.read();
      let sum = 0, n = 0;
      for (let p = 0; p < WIDTH * HEIGHT; p++) {
        if (!ball[p]) continue;
        for (let c = 0; c < 3; c++) { const d = Math.min(Math.max(img[3 * p + c], 0), 1) - truth[3 * p + c]; sum += d * d; n++; }
      }
      return -10 * Math.log10(sum / n);
    });
    const id = truthCanvas.createImageData(WIDTH, HEIGHT);
    for (let p = 0, q = 0; p < truth.length; p += 3, q += 4) { id.data[q] = truth[p] * 255; id.data[q + 1] = truth[p + 1] * 255; id.data[q + 2] = truth[p + 2] * 255; id.data[q + 3] = 255; }
    truthCanvas.putImageData(id, 0, 0);
    const [vx, vy, vw, vh] = view();
    const sx = WIDTH / vw;
    ctxs.forEach((c, i) => { c.imageSmoothingEnabled = !state.zoom; c.fillStyle = "#000"; c.fillRect(0, 0, WIDTH, HEIGHT); c.drawImage(off[i].canvas, vx, vy, vw, vh, 0, 0, WIDTH, HEIGHT); });
    const T = (x, y) => [(x - vx) * sx, (y - vy) * sx];
    ovs.forEach((o, i) => {
      o.clearRect(0, 0, WIDTH, HEIGHT);
      if (!state.paths) return;
      // the true path
      o.setLineDash([5, 4]); o.strokeStyle = "rgba(255,255,255,0.9)"; o.lineWidth = 1.5;
      o.beginPath();
      for (let f = 0; f <= 27; f += 0.25) { const [x, y] = T(...arcCentre(state.angle, f)); f ? o.lineTo(x, y) : o.moveTo(x, y); }
      o.stroke(); o.setLineDash([]);
      if (i === 2) return;
      // blobs on the ball: centre within the ball at their peak time, and living long enough to have a path
      const m = ms[i];
      o.strokeStyle = i === 0 ? "#cfcfcf" : "#7fb6ff"; o.lineWidth = 2;
      for (let j = 0; j < m.n; j++) {
        const t0 = m.t0[j], sg = m.sigma_t[j];
        if (sg < 0.5 || t0 < 0 || t0 > 9) continue;
        const [px, py] = centreAt(m, j, t0), [tx, ty] = arcCentre(state.angle, ARC.stride * t0);
        if (Math.hypot(px - tx, py - ty) > ARC.radius + 3) continue;
        const a = Math.max(0, t0 - 1.5 * sg), b = Math.min(9, t0 + 1.5 * sg);
        o.beginPath();
        for (let q = 0; q <= 30; q++) { const [x, y] = T(...centreAt(m, j, a + ((b - a) * q) / 30)); q ? o.lineTo(x, y) : o.moveTo(x, y); }
        o.stroke();
      }
    });
    psnrs.forEach((p, i) => { $(`[data-psnr="${i}"]`, root).textContent = `ball ${p.toFixed(2)} dB`; });
    $("#arc-status").textContent = `M0: ${ms[0].n.toLocaleString("en")} blobs, M1: ${ms[1].n.toLocaleString("en")} blobs · seed 0 · PSNR inside the ball at this time, computed in the browser`;
  }
  $("#arc-angle").oninput = () => { state.angle = angles[+$("#arc-angle").value]; render(); };
  $("#arc-t").oninput = () => { state.t = +$("#arc-t").value; render(); };
  segment($("#arc-budget"), null, "24000", (v) => { state.budget = +v; render(); });
  $("#arc-paths").onchange = () => { state.paths = $("#arc-paths").checked; render(); };
  $("#arc-zoom").onchange = () => { state.zoom = $("#arc-zoom").checked; render(); };
  let timer = null;
  $("#arc-play").onclick = () => {
    if (timer) { clearInterval(timer); timer = null; $("#arc-play").textContent = "Play"; return; }
    $("#arc-play").textContent = "Pause";
    timer = setInterval(() => { state.t = state.t + 1 / 30 > 9 ? 0 : state.t + 1 / 30; $("#arc-t").value = state.t; render(); }, 50);
  };
  render();
}

// ===================================================================================== table of contents
function setupToc() {
  const links = $$("#toc a");
  const sections = links.map((a) => $(a.getAttribute("href")));
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
  }, { rootMargin: "-40% 0px -55% 0px" });
  sections.forEach((s) => s && io.observe(s));
}

setupToc();
blobWidget();
budgetWidget();
fillTables();
trainingWidget();
playerWidget();
arcWidget();
