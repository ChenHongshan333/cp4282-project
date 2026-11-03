// Compares the renderer of the page (gauss.js, CPU path) with reference renders of the Python renderer
// written by `python -m stg.export_site --check --check-dir <folder>`.
//
//     node docs/assets/js/check.mjs <folder>

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseModel, sliceAt, renderCPU, arcFrame, WIDTH, HEIGHT } from "./gauss.js";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = join(here, "..", "..", "data");
const refDir = process.argv[2];
const { order, models } = JSON.parse(readFileSync(join(dataDir, "models.json"), "utf8"));
const index = JSON.parse(readFileSync(join(refDir, "index.json"), "utf8"));

const readFloats = (path) => { const b = readFileSync(path); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
let worst = 0;
for (const ref of index) {
  const expected = readFloats(join(refDir, ref.file));
  let image;
  if (ref.model) {
    const info = models[ref.model];
    const b = readFileSync(join(dataDir, info.file));
    const g = parseModel(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), info, order);
    image = renderCPU(sliceAt(g, ref.t), new Float32Array(WIDTH * HEIGHT * 3));
  } else {
    image = arcFrame(Number(ref.arc.slice(4)), ref.t);
  }
  let maxDiff = 0, sum = 0;
  for (let p = 0; p < image.length; p++) {
    const d = Math.abs(image[p] - expected[p]);
    maxDiff = Math.max(maxDiff, d);
    const c = Math.min(Math.max(image[p], 0), 1) - Math.min(Math.max(expected[p], 0), 1);
    sum += c * c;
  }
  const agreement = sum === 0 ? Infinity : -10 * Math.log10(sum / image.length);
  worst = Math.max(worst, maxDiff);
  console.log(`${(ref.model || ref.arc + " truth").padEnd(40)} t=${ref.t.toFixed(4)}  max |diff| ${maxDiff.toExponential(2)}  PSNR between the two ${agreement.toFixed(1)} dB`);
}
console.log(`largest pixel difference: ${worst.toExponential(2)}`);
