// One space-time Gaussian shown in the volume of the video (x, y, time), after the figure of the Unit 16
// notes ("The frame at time t is a slice"), for the four models: the centre follows a line (M0, M2)
// or a curve (M1, M3), and the slice keeps its size (M0, M1) or shrinks as it fades (M2, M3).
// Slicing follows stg/slicing.py: centre mu + v tau (+ a tau^2), Cholesky factor sqrt(s) L with
// s = max(w, s_min), weight w = exp(-tau^2 / (2 sigma_t^2)), skipped where |tau| >= 3.5 sigma_t.
//
// A page may hold several copies. Each finds its parts by class inside its root element: the
// figures .vol-svg, .vol-slice, .vol-weight, the sliders [data-k] with outputs [data-out], and
// optionally the model buttons .vol-model. A copy without model buttons shows M0 only.

import { WIDTH, HEIGHT, CUT, renderCPU } from "./gauss.js";

const FRAMES = 30, T0 = 15;
const CHOL = [42, -16, 17];                      // the Cholesky factor L of the slice, in pixels
const MODELS = { M0: [false, false], M1: [true, false], M2: [false, true], M3: [true, true] };   // [curve, shrink]
const BLUE = "#2a78d6", ORANGE = "#d4521d", INK = "#1b1b1b";

export function volumeWidget(root) {
  if (!root) return;
  const $ = (s) => root.querySelector(s);
  const svg = $(".vol-svg"), weightSvg = $(".vol-weight"), canvas = $(".vol-slice");
  const ctx = canvas.getContext("2d");
  const image = new Float32Array(WIDTH * HEIGHT * 3);
  const input = Object.fromEntries([...root.querySelectorAll("[data-k]")].map((el) => [el.dataset.k, el]));
  const value = (k, fallback) => (input[k] ? +input[k].value : fallback);
  const show = (k, text) => { const el = $(`[data-out="${k}"]`); if (el) el.textContent = text; };
  let model = "M0";

  const params = () => {
    const [curve, shrink] = MODELS[model];
    return { t: value("frame", T0), vx: value("vx", 2), vy: value("vy", -0.5),
             ax: curve ? value("ax", 0) : 0, ay: curve ? value("ay", 0) : 0,
             sigma: value("sigma", 5), smin: value("smin", 0.1), yaw: value("yaw", 35), curve, shrink };
  };
  const centre = (p, t) => { const tau = t - T0; return [WIDTH / 2 + p.vx * tau + p.ax * tau * tau, HEIGHT / 2 + p.vy * tau + p.ay * tau * tau]; };
  const weight = (p, t) => Math.exp(-0.5 * ((t - T0) / p.sigma) ** 2);
  const scale = (p, t) => (p.shrink ? Math.max(weight(p, t), p.smin) : 1);
  const visible = (p, t) => Math.abs(t - T0) < CUT * p.sigma;
  // the 2-sigma ellipse of the slice at time t, as image points
  const ellipse = (p, t, n = 40) => {
    const [cx, cy] = centre(p, t), r = Math.sqrt(scale(p, t));
    const [l11, l21, l22] = CHOL.map((v) => v * r);
    const pts = [];
    for (let k = 0; k <= n; k++) {
      const a = (2 * Math.PI * k) / n, z1 = 2 * Math.cos(a), z2 = 2 * Math.sin(a);
      pts.push([cx + l11 * z1, cy + l21 * z1 + l22 * z2]);
    }
    return pts;
  };

  // --- the volume: time along U, image x along V (depth), image y downwards; turned by the view angle
  function projector(yaw) {
    const th = (yaw * Math.PI) / 180, ph = (16 * Math.PI) / 180;
    const LU = 520, LV = 300, LZ = (LV * HEIGHT) / WIDTH;
    const raw = (t, x, y) => {
      const U = ((t - 1) / (FRAMES - 1) - 0.5) * LU, V = (x / WIDTH - 0.5) * LV, Z = (0.5 - y / HEIGHT) * LZ;
      const X1 = U * Math.cos(th) + V * Math.sin(th), D = -U * Math.sin(th) + V * Math.cos(th);
      return [X1, -(Z * Math.cos(ph) - D * Math.sin(ph))];
    };
    // fit the box into the view
    const corners = [];
    for (const t of [1, FRAMES]) for (const x of [0, WIDTH]) for (const y of [0, HEIGHT]) corners.push(raw(t, x, y));
    const xs = corners.map((c) => c[0]), ys = corners.map((c) => c[1]);
    const [W, H, m] = [600, 300, 26];
    const k = Math.min((W - 2 * m) / (Math.max(...xs) - Math.min(...xs)), (H - 2 * m) / (Math.max(...ys) - Math.min(...ys)));
    const ox = W / 2 - (k * (Math.max(...xs) + Math.min(...xs))) / 2, oy = H / 2 - (k * (Math.max(...ys) + Math.min(...ys))) / 2;
    return (t, x, y) => { const [a, b] = raw(t, x, y); return [ox + k * a, oy + k * b]; };
  }
  const path = (pts) => pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("");

  function drawVolume(p) {
    const P = projector(p.yaw);
    let s = "";
    // the box of the video
    const edge = (a, b) => `<path d="${path([P(...a), P(...b)])}" stroke="#cfcdc7" stroke-width="1" fill="none"/>`;
    for (const x of [0, WIDTH]) for (const y of [0, HEIGHT]) s += edge([1, x, y], [FRAMES, x, y]);
    for (const t of [1, FRAMES]) {
      s += edge([t, 0, 0], [t, WIDTH, 0]) + edge([t, 0, HEIGHT], [t, WIDTH, HEIGHT]) + edge([t, 0, 0], [t, 0, HEIGHT]) + edge([t, WIDTH, 0], [t, WIDTH, HEIGHT]);
    }
    // the 2-sigma ellipse of the slice, every half frame while the Gaussian is visible
    for (let t = 1; t <= FRAMES; t += 0.5) {
      if (!visible(p, t)) continue;
      const w = weight(p, t);
      s += `<path d="${path(ellipse(p, t).map(([x, y]) => P(t, x, y)))}Z" fill="none" stroke="${BLUE}" stroke-width="1" stroke-opacity="${(0.2 + 0.7 * w).toFixed(2)}"/>`;
    }
    // the path of the centre, a little beyond the clip
    const pts = [];
    for (let t = -2; t <= FRAMES + 3; t += 0.25) pts.push(P(t, ...centre(p, t)));
    s += `<path d="${path(pts)}" fill="none" stroke="${INK}" stroke-width="1.4" stroke-dasharray="5 4"/>`;
    // the current frame and its cut through the Gaussian
    const t = p.t;
    s += `<path d="${path([P(t, 0, 0), P(t, WIDTH, 0), P(t, WIDTH, HEIGHT), P(t, 0, HEIGHT)])}Z" fill="${ORANGE}" fill-opacity="0.09" stroke="${ORANGE}" stroke-width="1.5"/>`;
    if (visible(p, t)) {
      s += `<path d="${path(ellipse(p, t).map(([x, y]) => P(t, x, y)))}Z" fill="${ORANGE}" fill-opacity="${(0.15 + 0.5 * weight(p, t)).toFixed(2)}" stroke="${ORANGE}" stroke-width="2"/>`;
    }
    const [dx, dy] = P(t, ...centre(p, t));
    s += `<circle cx="${dx}" cy="${dy}" r="3" fill="${INK}"/>`;
    const [lx, ly] = P(t, 0, 0);
    s += `<text x="${lx}" y="${ly - 8}" fill="${ORANGE}" font-weight="600" text-anchor="middle">frame ${t}</text>`;
    const [ax, ay] = P(1, WIDTH, HEIGHT), [bx, by] = P(FRAMES, WIDTH, HEIGHT);
    s += `<text x="${ax}" y="${ay + 16}" text-anchor="middle">frame 1</text><text x="${bx}" y="${by + 16}" text-anchor="end">frame ${FRAMES}</text>`;
    svg.innerHTML = s;
  }

  // --- the slice at the current frame: the 2D Gaussian, drawn by the renderer of the models
  function drawSlice(p) {
    const t = p.t, on = visible(p, t), w = on ? weight(p, t) : 0, r = Math.sqrt(scale(p, t));
    const [cx, cy] = centre(p, t);
    const slice = { count: on ? 1 : 0, mu: new Float32Array([cx, cy]), chol: new Float32Array(CHOL.map((v) => v * r)),
                    colour: new Float32Array([w, w, w]) };
    renderCPU(slice, image);
    const out = ctx.createImageData(WIDTH, HEIGHT);
    const bg = [1, 1, 1], blue = [0.165, 0.47, 0.84];
    for (let p3 = 0, q = 0; p3 < image.length; p3 += 3, q += 4) {
      const a = Math.min(image[p3], 1);
      for (let c = 0; c < 3; c++) out.data[q + c] = Math.round(255 * (bg[c] * (1 - a) + blue[c] * a));
      out.data[q + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    ctx.setLineDash([6, 5]); ctx.strokeStyle = INK; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let s = T0 - 18, i = 0; s <= T0 + 18; s += 0.25, i++) { const [x, y] = centre(p, s); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke(); ctx.setLineDash([]);
    if (on) {
      ctx.strokeStyle = ORANGE; ctx.lineWidth = 3;
      ctx.beginPath();
      ellipse(p, t).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.stroke();
    }
    ctx.fillStyle = INK; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI); ctx.fill();
    $(".vol-frame-label").textContent = on ? `Frame ${t}: the slice is a 2D Gaussian` : `Frame ${t}: the Gaussian is not visible`;
  }

  // --- the weight of the slice in each frame, and with shrinking its size scale
  function drawWeight(p) {
    const W = 360, H = 150, L = 8, R = 8, T = 10, B = 26, bw = (W - L - R) / FRAMES;
    let s = `<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="#8f8d86"/>`;
    for (let k = 1; k <= FRAMES; k++) {
      const w = visible(p, k) ? weight(p, k) : 0, h = (H - T - B) * w, x = L + (k - 1) * bw;
      s += `<rect x="${(x + 1).toFixed(1)}" y="${(H - B - h).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${h.toFixed(1)}" fill="${k === p.t ? ORANGE : "#a9c6ef"}"/>`;
    }
    if (p.shrink) {
      const pts = [];
      for (let t = 1; t <= FRAMES; t += 0.25) if (visible(p, t)) pts.push([L + (t - 0.5) * bw, H - B - (H - T - B) * scale(p, t)]);
      s += `<path d="${path(pts)}" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
    }
    s += `<text x="${L}" y="${H - 8}">1</text><text x="${W / 2}" y="${H - 8}" text-anchor="middle">frame</text><text x="${W - R}" y="${H - 8}" text-anchor="end">${FRAMES}</text>`;
    weightSvg.innerHTML = s;
    const key = $(".vol-weight-key");
    if (key) key.style.display = p.shrink ? "" : "none";
  }

  function draw() {
    const p = params();
    show("frame", `${p.t} of ${FRAMES}`);
    show("vx", p.vx.toFixed(1));
    show("vy", p.vy.toFixed(1));
    show("ax", p.curve ? value("ax", 0).toFixed(2) : "off");
    show("ay", p.curve ? value("ay", 0).toFixed(2) : "off");
    show("sigma", p.sigma.toFixed(1));
    show("smin", p.shrink ? p.smin.toFixed(2) : "off");
    show("yaw", `${p.yaw}°`);
    if (input.ax) input.ax.disabled = input.ay.disabled = !p.curve;
    if (input.smin) input.smin.disabled = !p.shrink;
    const cost = $(".vol-cost");
    if (cost) cost.textContent = p.curve ? "14 numbers per Gaussian (12 + the acceleration a)" : "12 numbers per Gaussian";
    drawVolume(p); drawSlice(p); drawWeight(p);
  }

  const seg = $(".vol-model");
  if (seg) {
    seg.onclick = (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      model = b.dataset.v;
      seg.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      draw();
    };
  }
  Object.values(input).forEach((el) => el.addEventListener("input", draw));
  draw();
}
