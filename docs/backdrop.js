/* Landing backdrop: a slow energy ribbon.
 *
 * The motif is the product's own -- the energy meter and the timeline chart are
 * both bars under a cyan -> amber -> hot pink ramp -- so the page moves in the
 * same language it measures in. It is deliberately abstract rather than a
 * mock skeleton overlay: this project's whole argument is about not dressing
 * invented data up as measurement, and that has to hold on the landing page too.
 *
 * Two rules it must obey:
 *   - it stops dead when the camera starts, because inference needs the frame
 *     budget far more than a background does;
 *   - it never runs for someone who asked for reduced motion.
 */

const COLD = [34, 211, 238], WARM = [251, 191, 36], HOT = [244, 63, 94];
const BARS = 96;
const FPS = 24;

function ramp(t) {
  const u = Math.max(0, Math.min(1, t));
  const [a, b, k] = u < 0.5 ? [COLD, WARM, u / 0.5] : [WARM, HOT, (u - 0.5) / 0.5];
  return a.map((v, i) => Math.round(v + (b[i] - v) * k));
}

export class Backdrop {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.raf = null;
    this.last = 0;
    this.t = 0;
    this.reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this._resize = () => this.resize();
  }

  resize() {
    const r = this.cv.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);   // 3x DPR buys nothing here
    this.cv.width = Math.max(1, Math.round(r.width * dpr));
    this.cv.height = Math.max(1, Math.round(r.height * dpr));
    this.dpr = dpr;
  }

  start() {
    this.resize();
    addEventListener("resize", this._resize);
    if (this.reduced) { this.draw(); return; }   // one static frame, then nothing
    const loop = (now) => {
      this.raf = requestAnimationFrame(loop);
      if (now - this.last < 1000 / FPS) return;
      this.last = now;
      this.t += 0.016;
      this.draw();
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop() {
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
    removeEventListener("resize", this._resize);
    this.ctx.clearRect(0, 0, this.cv.width, this.cv.height);
  }

  draw() {
    const { ctx, cv } = this;
    const W = cv.width, H = cv.height, t = this.t;
    ctx.clearRect(0, 0, W, H);

    const gap = W / BARS;
    const bw = gap * 0.34;
    const baseline = H * 0.86;
    const maxH = H * 0.38;

    // A playhead sweeps the ribbon on a slow loop. Without it the bars breathe
    // so gently that the page reads as static; with it the motion is obvious
    // but still peripheral, and it borrows the one gesture every DJ already
    // reads instantly -- a position marker crossing a waveform.
    const head = (t * 0.075) % 1.35 - 0.175;   // off-canvas either side

    for (let i = 0; i < BARS; i++) {
      const x = i * gap + gap / 2;
      const p = i / BARS;
      // Three incommensurate sines: the pattern never visibly repeats, and it
      // costs nothing next to real noise.
      const v =
        0.5 +
        0.26 * Math.sin(p * 7.1 + t * 0.9) +
        0.16 * Math.sin(p * 15.7 - t * 1.37) +
        0.10 * Math.sin(p * 3.3 + t * 0.53);
      // Gaussian falloff around the head, so it lifts a region rather than a
      // hard edge of bars.
      const d = (p - head) / 0.11;
      const lift = Math.exp(-d * d);
      const h = Math.max(2, v * maxH * (1 + 0.55 * lift));
      // Fade the ends so the ribbon dissolves instead of being cropped.
      const edge = Math.min(1, Math.min(p, 1 - p) / 0.14);
      const [r, g, b] = ramp(Math.min(1, v + 0.3 * lift));
      ctx.fillStyle = `rgba(${r},${g},${b},${(0.055 + 0.16 * lift) * edge})`;
      ctx.fillRect(x - bw / 2, baseline - h, bw, h);
      // A dim mirrored stub below the baseline: reads as a reflection on glass.
      ctx.fillStyle = `rgba(${r},${g},${b},${(0.02 + 0.05 * lift) * edge})`;
      ctx.fillRect(x - bw / 2, baseline, bw, h * 0.34);
    }
  }
}
