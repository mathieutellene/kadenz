/* Vision pipeline, browser edition.
 *
 * Same shape as the desktop engine, different machinery:
 *
 *   desktop                          browser
 *   -------                          -------
 *   Ultralytics YOLOv8-pose (torch)  the same weights, exported to ONNX and run
 *                                    by onnxruntime-web (WebGPU, WASM fallback)
 *   ByteTrack (supervision)          greedy IoU tracker, below
 *   Farneback dense optical flow     per-box frame differencing
 *
 * The tracker and the motion estimate are deliberately simpler than the desktop
 * ones -- this has to hold a frame budget inside a browser tab. What is NOT
 * simplified is the part the project is actually about: the energy signal keeps
 * the same session-relative percentile normalisation, so the reward handed to
 * the recommender means the same thing here as it does in Python.
 */

/* The ONNX graphs are exported at fixed input sizes, so the size is a property
   of whichever model was loaded rather than a global. 320 is the default because
   it is the only one that holds a usable frame rate on single-threaded WASM --
   GitHub Pages cannot send the COOP/COEP headers that would unlock WASM threads,
   so that is the floor everyone lands on without WebGPU. 640 is the same dial as
   `det_imgsz` on the desktop and is fetched only if the visitor asks for it. */
export const MODELS = {
  320: { url: "model/pose320.onnx", label: "fast" },
  640: { url: "model/pose640.onnx", label: "crowd mode" },
};
const CONF = 0.35;
const IOU_NMS = 0.45;
const KP_CONF = 0.45;

/* COCO-17 skeleton, grouped so a missing limb never links unrelated joints. */
export const EDGES = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 5], [0, 6],
];

/* ------------------------------------------------------------------ model */
export async function loadModel(size, onStatus) {
  const url = MODELS[size].url;
  const ort = window.ort;
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
  ort.env.wasm.simd = true;

  // WebGPU is roughly an order of magnitude faster where it exists, but it is
  // still absent or broken in plenty of browsers, so never hard-depend on it.
  const tries = [];
  if ("gpu" in navigator) tries.push("webgpu");
  tries.push("wasm");

  let lastErr;
  for (const ep of tries) {
    try {
      onStatus?.(`loading model (${ep})`);
      const session = await ort.InferenceSession.create(url, {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      return { session, backend: ep, size };
    } catch (e) {
      lastErr = e;
      console.warn(`[kadenz] ${ep} unavailable:`, e.message);
    }
  }
  throw lastErr ?? new Error("no execution provider available");
}

/* --------------------------------------------------------------- pre/post */
/* Letterbox into a square without distorting people: aspect ratio matters to a
   pose model, and squashing the frame measurably costs keypoints. */
export function makeLetterbox(srcW, srcH, size) {
  const scale = Math.min(size / srcW, size / srcH);
  const w = Math.round(srcW * scale);
  const h = Math.round(srcH * scale);
  return { scale, w, h, size, dx: Math.floor((size - w) / 2), dy: Math.floor((size - h) / 2) };
}

export function preprocess(ctx, size, buffer) {
  const { data } = ctx.getImageData(0, 0, size, size);
  const plane = size * size;
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    buffer[i] = data[p] / 255;                 // R
    buffer[i + plane] = data[p + 1] / 255;     // G
    buffer[i + plane * 2] = data[p + 2] / 255; // B
  }
  return buffer;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

/* YOLOv8-pose emits [1, 56, 8400]: 4 box + 1 score + 17 keypoints x (x, y, c),
   laid out attribute-major, so anchor i of attribute a lives at a * 8400 + i. */
export function postprocess(out, lb, srcW, srcH) {
  const A = out.dims[2];
  const d = out.data;
  const dets = [];
  for (let i = 0; i < A; i++) {
    const score = d[4 * A + i];
    if (score < CONF) continue;
    const cx = d[i], cy = d[A + i], w = d[2 * A + i], h = d[3 * A + i];
    const un = (v, off) => (v - off) / lb.scale;
    const x1 = un(cx - w / 2, lb.dx), y1 = un(cy - h / 2, lb.dy);
    const x2 = un(cx + w / 2, lb.dx), y2 = un(cy + h / 2, lb.dy);
    const kpts = new Float32Array(34);
    const kconf = new Float32Array(17);
    for (let k = 0; k < 17; k++) {
      const base = (5 + k * 3) * A + i;
      kpts[k * 2] = un(d[base], lb.dx);
      kpts[k * 2 + 1] = un(d[base + A], lb.dy);
      kconf[k] = d[base + 2 * A];
    }
    dets.push({
      score,
      x1: Math.max(0, x1), y1: Math.max(0, y1),
      x2: Math.min(srcW, x2), y2: Math.min(srcH, y2),
      kpts, kconf,
    });
  }
  dets.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const det of dets) {
    if (keep.every((k) => iou(k, det) < IOU_NMS)) keep.push(det);
    if (keep.length >= 60) break;
  }
  return keep;
}

/* ---------------------------------------------------------------- tracker */
/* Greedy IoU association. ByteTrack's two-stage matching buys robustness under
   occlusion that matters on a packed floor and much less on a webcam, so this
   trades it for a pipeline that fits in a frame budget. */
export class Tracker {
  constructor({ iouMin = 0.3, maxAge = 12, minHits = 3, holdAge = 2 } = {}) {
    Object.assign(this, { iouMin, maxAge, minHits, holdAge });
    this.tracks = [];
    this.nextId = 1;
    this.entries = 0;
    this.exits = 0;
  }

  update(dets) {
    for (const t of this.tracks) t.matched = false;
    const used = new Set();

    const pairs = [];
    this.tracks.forEach((t, ti) => {
      dets.forEach((d, di) => {
        const v = iou(t, d);
        if (v >= this.iouMin) pairs.push({ ti, di, v });
      });
    });
    pairs.sort((a, b) => b.v - a.v);
    for (const { ti, di } of pairs) {
      const t = this.tracks[ti];
      if (t.matched || used.has(di)) continue;
      const d = dets[di];
      Object.assign(t, {
        x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2,
        kpts: d.kpts, kconf: d.kconf,
        matched: true, age: 0, hits: t.hits + 1,
      });
      used.add(di);
    }

    dets.forEach((d, di) => {
      if (used.has(di)) return;
      this.tracks.push({
        id: this.nextId++, ...d, matched: true, age: 0, hits: 1,
        raw: 0, e: null, counted: false,
      });
    });

    const alive = [];
    for (const t of this.tracks) {
      if (!t.matched) t.age += 1;
      if (t.age > this.maxAge) {
        if (t.counted) this.exits += 1;     // only confirmed tracks count as exits
        continue;
      }
      if (!t.counted && t.hits >= this.minHits) { t.counted = true; this.entries += 1; }
      alive.push(t);
    }
    this.tracks = alive;
    // age <= holdAge, not age === 0: a confirmed person missed for a frame or
    // two would otherwise blink out of the overlay entirely. Slightly stale
    // geometry beats a strobing skeleton.
    return this.tracks.filter((t) => t.hits >= this.minHits && t.age <= this.holdAge);
  }
}

/* ------------------------------------------------------------ motion energy */
/* Frame differencing on a small grayscale buffer. Farneback gives a true flow
   field; this gives "how much of this person's pixels changed", which is a
   coarser but serviceable stand-in and costs almost nothing. */
export class Motion {
  constructor(w = 192, h = 108) {
    this.w = w; this.h = h;
    this.prev = null;
    this.cur = new Float32Array(w * h);
    this.canvas = Object.assign(document.createElement("canvas"), { width: w, height: h });
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  ingest(video) {
    this.ctx.drawImage(video, 0, 0, this.w, this.h);
    const { data } = this.ctx.getImageData(0, 0, this.w, this.h);
    for (let i = 0, p = 0; i < this.cur.length; i++, p += 4) {
      this.cur[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    }
    const prev = this.prev;
    this.prev = Float32Array.from(this.cur);
    if (!prev) return null;
    // Strobes and auto-exposure jumps shift the whole frame at once; a global
    // brightness delta is not dancing, so gate those frames out entirely.
    let sum = 0;
    for (let i = 0; i < this.cur.length; i++) sum += this.cur[i] - prev[i];
    if (Math.abs(sum / this.cur.length) > 12) return null;
    const diff = new Float32Array(this.cur.length);
    for (let i = 0; i < diff.length; i++) diff[i] = Math.abs(this.cur[i] - prev[i]);
    return diff;
  }

  /* Mean absolute change inside a box, in source-frame coordinates. */
  inBox(diff, box, srcW, srcH) {
    if (!diff) return 0;
    const sx = this.w / srcW, sy = this.h / srcH;
    const x1 = Math.max(0, Math.floor(box.x1 * sx)), x2 = Math.min(this.w, Math.ceil(box.x2 * sx));
    const y1 = Math.max(0, Math.floor(box.y1 * sy)), y2 = Math.min(this.h, Math.ceil(box.y2 * sy));
    if (x2 <= x1 || y2 <= y1) return 0;
    let s = 0, n = 0;
    for (let y = y1; y < y2; y++) {
      const row = y * this.w;
      for (let x = x1; x < x2; x++) { s += diff[row + x]; n++; }
    }
    return n ? s / n : 0;
  }
}

/* Session-relative percentile scale, matching engine/metrics.py: an absolute
   motion number is meaningless across rooms, cameras and lighting, so 0-100
   always means "against what THIS session has seen". */
export class Normaliser {
  constructor(cap = 1200) { this.cap = cap; this.hist = []; }

  add(v) {
    this.hist.push(v);
    if (this.hist.length > this.cap) this.hist.shift();
  }

  rank(v) {
    const n = this.hist.length;
    if (n < 30) return null;             // refuse to score before calibrating
    let below = 0;
    for (const h of this.hist) if (h < v) below++;
    return Math.max(0, Math.min(100, (100 * below) / n));
  }
}
