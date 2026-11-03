// The renderer of gauss.js on the GPU with WebGL2: one instanced quad per 2D Gaussian over the box of
// the course kernel, additive blending into a float texture, then clamping to [0, 1] for display.
// The per-pixel arithmetic is the course's; float32 throughout, as in the Warp kernel.

import { WIDTH, HEIGHT, CUT, renderCPU } from "./gauss.js";

const VERT = `#version 300 es
layout(location = 0) in vec2 corner;
layout(location = 1) in vec2 mu;
layout(location = 2) in vec3 chol;
layout(location = 3) in vec3 colour;
uniform vec2 size;
uniform float cut;
flat out vec2 vMu;
flat out vec3 vChol;
flat out vec3 vColour;
void main() {
  vec2 r = cut * vec2(abs(chol.x), sqrt(chol.y * chol.y + chol.z * chol.z)) + 1.0;
  vec2 p = mu + corner * r;                          // pixel coordinates, y down
  gl_Position = vec4(p.x / size.x * 2.0 - 1.0, 1.0 - p.y / size.y * 2.0, 0.0, 1.0);
  vMu = mu; vChol = chol; vColour = colour;
}`;

const FRAG = `#version 300 es
precision highp float;
uniform vec2 size;
uniform float cut;
flat in vec2 vMu;
flat in vec3 vChol;
flat in vec3 vColour;
out vec4 outColour;
void main() {
  float x = floor(gl_FragCoord.x), y = size.y - 1.0 - floor(gl_FragCoord.y);
  float z1 = (x + 0.5 - vMu.x) / vChol.x;
  float z2 = (y + 0.5 - vMu.y - vChol.y * z1) / vChol.z;
  float q = z1 * z1 + z2 * z2;
  if (!(q < cut * cut)) discard;
  outColour = vec4(exp(-0.5 * q) * vColour, 1.0);
}`;

// Display: the accumulated image clamped to [0, 1], or the error |image - target| x gain.
const SHOW_VERT = `#version 300 es
layout(location = 0) in vec2 corner;
out vec2 uv;
void main() { uv = corner * 0.5 + 0.5; gl_Position = vec4(corner, 0.0, 1.0); }`;

const SHOW_FRAG = `#version 300 es
precision highp float;
uniform sampler2D image;
uniform sampler2D target;
uniform int mode;
uniform float gain;
in vec2 uv;
out vec4 outColour;
void main() {
  vec3 c = clamp(texture(image, uv).rgb, 0.0, 1.0);
  if (mode == 1) {
    vec3 t = texture(target, vec2(uv.x, 1.0 - uv.y)).rgb;
    c = clamp(abs(c - t) * gain, 0.0, 1.0);
  }
  outColour = vec4(c, 1.0);
}`;

function compile(gl, vert, frag) {
  const program = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vert], [gl.FRAGMENT_SHADER, frag]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(program, s);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}

export class GLRenderer {
  // Renders into its own canvas (width x height); copy the result with drawImage(renderer.canvas, ...).
  constructor(width = WIDTH, height = HEIGHT) {
    this.width = width; this.height = height;
    this.canvas = document.createElement("canvas");
    this.canvas.width = width; this.canvas.height = height;
    const gl = this.canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    const float32 = gl.getExtension("EXT_color_buffer_float") && gl.getExtension("EXT_float_blend");
    const half = float32 || gl.getExtension("EXT_color_buffer_half_float") || gl.getExtension("EXT_color_buffer_float");
    if (!half) throw new Error("float render targets are not available");
    this.precision = float32 ? "float32" : "float16";

    this.splat = compile(gl, VERT, FRAG);
    this.show = compile(gl, SHOW_VERT, SHOW_FRAG);
    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, float32 ? gl.RGBA32F : gl.RGBA16F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
    this.target = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.target);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.buffers = [];
    for (const [loc, n] of [[1, 2], [2, 3], [3, 3]]) {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
      this.buffers.push(b);
    }
  }

  // Accumulate a slice (gauss.sliceAt) into the float texture.
  draw(slice) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.width, this.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (slice.count === 0) return;
    gl.useProgram(this.splat);
    gl.uniform2f(gl.getUniformLocation(this.splat, "size"), this.width, this.height);
    gl.uniform1f(gl.getUniformLocation(this.splat, "cut"), CUT);
    gl.bindVertexArray(this.vao);
    const data = [slice.mu.subarray(0, 2 * slice.count), slice.chol.subarray(0, 3 * slice.count),
                  slice.colour.subarray(0, 3 * slice.count)];
    data.forEach((d, i) => { gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers[i]); gl.bufferData(gl.ARRAY_BUFFER, d, gl.DYNAMIC_DRAW); });
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, slice.count);
    gl.disable(gl.BLEND);
  }

  // Show the texture on the canvas: mode "image", or "error" against the target set by setTarget.
  present(mode = "image", gain = 4) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(this.show);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(gl.getUniformLocation(this.show, "image"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.target);
    gl.uniform1i(gl.getUniformLocation(this.show, "target"), 1);
    gl.uniform1i(gl.getUniformLocation(this.show, "mode"), mode === "error" ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(this.show, "gain"), gain);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.activeTexture(gl.TEXTURE0);
  }

  setTarget(source) {              // an <img>, <canvas> or ImageData of width x height
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.target);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  }

  // The accumulated float image (H * W * 3, rows top to bottom), for checking against renderCPU.
  read() {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    const rgba = new Float32Array(this.width * this.height * 4);
    gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.FLOAT, rgba);
    const out = new Float32Array(this.width * this.height * 3);
    for (let y = 0; y < this.height; y++) {
      const src = (this.height - 1 - y) * this.width, dst = y * this.width;
      for (let x = 0; x < this.width; x++)
        for (let c = 0; c < 3; c++) out[3 * (dst + x) + c] = rgba[4 * (src + x) + c];
    }
    return out;
  }
}

// A renderer with the same interface on the CPU, for browsers without float render targets.
export class CPURenderer {
  constructor(width = WIDTH, height = HEIGHT) {
    this.width = width; this.height = height; this.precision = "cpu";
    this.canvas = document.createElement("canvas");
    this.canvas.width = width; this.canvas.height = height;
    this.ctx = this.canvas.getContext("2d");
    this.image = new Float32Array(width * height * 3);
    this.targetData = null;
  }
  draw(slice) { renderCPU(slice, this.image, this.width, this.height); }
  present(mode = "image", gain = 4) {
    const out = this.ctx.createImageData(this.width, this.height);
    const t = this.targetData;
    for (let p = 0, q = 0; p < this.image.length; p += 3, q += 4) {
      for (let c = 0; c < 3; c++) {
        let v = Math.min(Math.max(this.image[p + c], 0), 1);
        if (mode === "error" && t) v = Math.min(Math.abs(v - t[q + c] / 255) * gain, 1);
        out.data[q + c] = Math.round(v * 255);
      }
      out.data[q + 3] = 255;
    }
    this.ctx.putImageData(out, 0, 0);
  }
  setTarget(source) {
    const c = document.createElement("canvas");
    c.width = this.width; c.height = this.height;
    const ctx = c.getContext("2d");
    if (source instanceof ImageData) ctx.putImageData(source, 0, 0); else ctx.drawImage(source, 0, 0);
    this.targetData = ctx.getImageData(0, 0, this.width, this.height).data;
  }
  read() { return this.image.slice(); }
}

export function makeRenderer() {
  try { return new GLRenderer(); } catch (e) { console.warn("WebGL renderer unavailable, using the CPU:", e.message); return new CPURenderer(); }
}
