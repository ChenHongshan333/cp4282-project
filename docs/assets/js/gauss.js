// Space-time Gaussians in the browser: loading a fitted model, slicing it at a time t, and rendering
// the slice. The arithmetic follows stg/slicing.py and the course kernel (gaussian_image_video.py)
// line by line; docs/assets/js/check.mjs compares this renderer with the Python one.

export const WIDTH = 480, HEIGHT = 270;
export const CUT = 3.5;          // a Gaussian is evaluated where |z| < CUT, in space and in time
export const POLY_TIME = 4.5;    // the time unit of the powers above 2 (D3, D7)

export const SPECS = {
  M0: { accel: false, shrink: false, degree: 0 },
  M1: { accel: true, shrink: false, degree: 0 },
  M2: { accel: false, shrink: true, degree: 0 },
  M3: { accel: true, shrink: true, degree: 0 },
  D3: { accel: true, shrink: false, degree: 3 },
  D7: { accel: true, shrink: false, degree: 7 },
};

const extraPowers = (spec) => Math.max(spec.degree - 2, 0);

// Numbers stored per Gaussian (stg/model.py).
export function numbersPerGaussian(name) {
  const spec = SPECS[name];
  return 12 + (spec.accel ? 2 : 0) + 2 * extraPowers(spec);
}

// A model from the float32 file written by stg/export_site.py: the stored attributes one after another.
export function parseModel(buffer, info, order) {
  const spec = SPECS[info.model];
  const n = info.gaussians, k = extraPowers(spec);
  const sizes = { mu: 2, vel: 2, acc: spec.accel ? 2 : 0, poly: 2 * k, t0: 1, sigma_t: 1, chol: 3, colour: 3 };
  const all = new Float32Array(buffer);
  const model = { name: info.model, spec, n, s_min: info.s_min };
  let offset = 0;
  for (const attr of order) {
    const size = sizes[attr] * n;
    if (size > 0) model[attr] = all.subarray(offset, offset + size);
    offset += size;
  }
  if (offset !== all.length) throw new Error(`model ${info.model}: ${all.length} numbers, expected ${offset}`);
  return model;
}

// The 2D Gaussians of the model at clip time t: centres, Cholesky factors and time-weighted colours
// of the active ones, and the index of each in the model.
export function sliceAt(g, t) {
  const { spec, n } = g;
  const k = extraPowers(spec);
  const mu = new Float32Array(2 * n), chol = new Float32Array(3 * n), colour = new Float32Array(3 * n);
  const index = new Int32Array(n);
  let m = 0;
  for (let i = 0; i < n; i++) {
    const tau = t - g.t0[i], sigma = g.sigma_t[i];
    if (!(Math.abs(tau) < CUT * sigma)) continue;
    const w = Math.exp(-0.5 * (tau / sigma) ** 2);
    let x = g.mu[2 * i] + g.vel[2 * i] * tau, y = g.mu[2 * i + 1] + g.vel[2 * i + 1] * tau;
    if (spec.accel) {
      x += g.acc[2 * i] * tau * tau;
      y += g.acc[2 * i + 1] * tau * tau;
    }
    for (let j = 0; j < k; j++) {             // the power j + 3
      const f = tau * tau * (tau / POLY_TIME) ** (j + 1);
      x += g.poly[(i * k + j) * 2] * f;
      y += g.poly[(i * k + j) * 2 + 1] * f;
    }
    const r = spec.shrink ? Math.sqrt(Math.max(w, g.s_min)) : 1;
    mu[2 * m] = x; mu[2 * m + 1] = y;
    for (let c = 0; c < 3; c++) {
      chol[3 * m + c] = g.chol[3 * i + c] * r;
      colour[3 * m + c] = g.colour[3 * i + c] * w;
    }
    index[m++] = i;
  }
  return { count: m, mu, chol, colour, index };
}

// The accumulated sum of the course renderer, on the CPU, into image (H * W * 3, not clamped).
export function renderCPU(slice, image, width = WIDTH, height = HEIGHT) {
  image.fill(0);
  const cut2 = CUT * CUT;
  for (let i = 0; i < slice.count; i++) {
    const mx = slice.mu[2 * i], my = slice.mu[2 * i + 1];
    const l11 = slice.chol[3 * i], l21 = slice.chol[3 * i + 1], l22 = slice.chol[3 * i + 2];
    const rx = CUT * Math.abs(l11), ry = CUT * Math.sqrt(l21 * l21 + l22 * l22);
    const x0 = Math.max(Math.floor(mx - rx), 0), x1 = Math.min(Math.floor(mx + rx), width - 1);
    const y0 = Math.max(Math.floor(my - ry), 0), y1 = Math.min(Math.floor(my + ry), height - 1);
    const cr = slice.colour[3 * i], cg = slice.colour[3 * i + 1], cb = slice.colour[3 * i + 2];
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const z1 = (x + 0.5 - mx) / l11;
        const z2 = (y + 0.5 - my - l21 * z1) / l22;
        const q = z1 * z1 + z2 * z2;
        if (q < cut2) {
          const w = Math.exp(-0.5 * q), p = 3 * (y * width + x);
          image[p] += w * cr; image[p + 1] += w * cg; image[p + 2] += w * cb;
        }
      }
    }
  }
  return image;
}

// Centre of a Gaussian at time t (for drawing paths).
export function centreAt(g, i, t) {
  const tau = t - g.t0[i], k = extraPowers(g.spec);
  let x = g.mu[2 * i] + g.vel[2 * i] * tau, y = g.mu[2 * i + 1] + g.vel[2 * i + 1] * tau;
  if (g.spec.accel) { x += g.acc[2 * i] * tau * tau; y += g.acc[2 * i + 1] * tau * tau; }
  for (let j = 0; j < k; j++) {
    const f = tau * tau * (tau / POLY_TIME) ** (j + 1);
    x += g.poly[(i * k + j) * 2] * f; y += g.poly[(i * k + j) * 2 + 1] * f;
  }
  return [x, y];
}

// --- the arc clips of plan 2e' (stg/data.py, style "arcs") ------------------------------------------

export const ARC = { speed: 3.0, radius: 8.0, period: 30.0, frames: 27, stride: 3 };

// Centre of the ball at a (fractional) frame of the sequence; angle in degrees turned over 27 frames.
export function arcCentre(angle, frame) {
  const length = ARC.speed * ARC.frames;
  const u = ARC.speed * frame - length / 2;
  let x, y, offset = 0;
  if (angle === 0) {
    x = u; y = 0;
  } else {
    const radius = length / (angle * Math.PI / 180);
    x = radius * Math.sin(u / radius);
    y = radius * (1 - Math.cos(u / radius));
    let lo = Infinity, hi = -Infinity;
    for (let s = 0; s < 200; s++) {            // np.linspace(-length / 2, length / 2, 200)
      const span = -length / 2 + (length * s) / 199;
      const v = radius * (1 - Math.cos(span / radius));
      lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    offset = (hi + lo) / 2;
  }
  return [x + WIDTH / 2, y + HEIGHT / 2 - offset];
}

// The arc clip at clip time t (frame 3t of the sequence), as float RGB (H * W * 3).
export function arcFrame(angle, t, image = new Float32Array(WIDTH * HEIGHT * 3)) {
  const [cx, cy] = arcCentre(angle, ARC.stride * t);
  for (let yi = 0; yi < HEIGHT; yi++) {
    const y = yi + 0.5;
    for (let xi = 0; xi < WIDTH; xi++) {
      const x = xi + 0.5, p = 3 * (yi * WIDTH + xi);
      let r = 0.3 + 0.3 * x / WIDTH, gr = 0.4 + 0.1 * Math.sin(2 * Math.PI * y / HEIGHT * 2), b = 0.5 - 0.2 * y / HEIGHT;
      const dx = x - cx, dy = y - cy, d = Math.sqrt(dx * dx + dy * dy);
      const alpha = Math.min(Math.max(ARC.radius + 1.0 - d, 0), 1);
      if (alpha > 0) {
        const s = 0.5 + 0.5 * Math.sin(2 * Math.PI * (dx + 0.5 * dy) / ARC.period);
        r = (1 - alpha) * r + alpha * (0.9 - 0.4 * s);
        gr = (1 - alpha) * gr + alpha * (0.2 + 0.6 * s);
        b = (1 - alpha) * b + alpha * 0.15;
      }
      image[p] = r; image[p + 1] = gr; image[p + 2] = b;
    }
  }
  return image;
}

// PSNR of an image against a target, both float RGB; the image is clamped to [0, 1] first.
export function psnr(image, target, mask = null) {
  let sum = 0, count = 0;
  for (let p = 0; p < image.length; p++) {
    if (mask && !mask[(p / 3) | 0]) continue;
    const d = Math.min(Math.max(image[p], 0), 1) - target[p];
    sum += d * d; count++;
  }
  return -10 * Math.log10(sum / count);
}
