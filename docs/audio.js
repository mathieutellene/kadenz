/* Audio analysis, browser edition.
 *
 * The desktop build runs librosa over a decoded waveform. In a tab there is no
 * librosa and no decoded file -- only a live AnalyserNode -- so the onset
 * envelope is spectral flux computed per animation frame, and tempo comes from
 * autocorrelating that envelope. Coarser than librosa, same shape of signal,
 * and enough for the one metric that needs it: Groove Sync.
 *
 * Nothing here fabricates a number from silence. Below the noise floor the
 * getters return null and the UI shows "--" rather than an invented BPM.
 */

const SILENCE_DBFS = -55;
const SPL_OFFSET = 108;        // matches config.yaml audio.spl_offset
const ENV_HZ = 20;             // envelope sample rate we resample onto
const ENV_SECONDS = 20;

export class AudioAnalyser {
  constructor() {
    this.ctx = null;
    this.analyser = null;
    this.freq = null;
    this.time = null;
    this.prevSpec = null;
    this.env = [];             // {t, onset}
    this.ready = false;
    this.error = null;
  }

  async start(stream) {
    const track = stream.getAudioTracks()[0];
    if (!track) { this.error = "no audio track"; return false; }
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    const src = this.ctx.createMediaStreamSource(stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.0;   // we want the raw flux
    src.connect(this.analyser);
    this.freq = new Uint8Array(this.analyser.frequencyBinCount);
    this.time = new Uint8Array(this.analyser.fftSize);
    this.ready = true;
    return true;
  }

  /* Call once per rendered frame. Returns {db, onset} or null when silent. */
  tick(t) {
    if (!this.ready) return null;
    this.analyser.getByteTimeDomainData(this.time);
    let sum = 0;
    for (let i = 0; i < this.time.length; i++) {
      const v = (this.time[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / this.time.length);
    const dbfs = 20 * Math.log10(Math.max(rms, 1e-9));
    if (dbfs < SILENCE_DBFS) return null;   // silence: do not invent anything

    this.analyser.getByteFrequencyData(this.freq);
    let flux = 0;
    if (this.prevSpec) {
      for (let i = 0; i < this.freq.length; i++) {
        const d = this.freq[i] - this.prevSpec[i];
        if (d > 0) flux += d;               // half-wave rectified: onsets only
      }
      flux /= this.freq.length;
    }
    this.prevSpec = Uint8Array.from(this.freq);

    const last = this.env[this.env.length - 1];
    if (!last || t - last.t >= 1 / ENV_HZ) {
      this.env.push({ t, onset: flux });
      const cutoff = t - ENV_SECONDS;
      while (this.env.length && this.env[0].t < cutoff) this.env.shift();
    }
    return { db: +(dbfs + SPL_OFFSET).toFixed(1), onset: flux };
  }

  /* Tempo by autocorrelating the onset envelope over 70-170 BPM. */
  bpm() {
    const n = this.env.length;
    if (n < ENV_HZ * 8) return null;        // need a few bars before guessing
    const x = this.env.map((e) => e.onset);
    const m = x.reduce((s, v) => s + v, 0) / n;
    const c = x.map((v) => v - m);
    let best = null, bestScore = 0;
    for (let bpm = 70; bpm <= 170; bpm += 0.5) {
      const lag = Math.round((60 / bpm) * ENV_HZ);
      if (lag < 2 || lag >= n - ENV_HZ) continue;
      let s = 0, k = 0;
      for (let i = 0; i + lag < n; i++, k++) s += c[i] * c[i + lag];
      if (!k) continue;
      const score = s / k;
      if (score > bestScore) { bestScore = score; best = bpm; }
    }
    return best == null || bestScore <= 0 ? null : Math.round(best);
  }

  /* The onset envelope resampled onto a fixed grid, for Groove Sync. */
  envelopeOn(grid) {
    if (this.env.length < 4) return null;
    return grid.map((t) => {
      let lo = 0, hi = this.env.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (this.env[mid].t < t) lo = mid + 1; else hi = mid;
      }
      return this.env[lo].onset;
    });
  }

  stop() { try { this.ctx?.close(); } catch { /* already gone */ } this.ready = false; }
}

/* --------------------------------------------------------------- groove sync */
function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 8) return null;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  if (da <= 0 || db <= 0) return null;
  return num / Math.sqrt(da * db);
}

const diff = (xs) => xs.slice(1).map((v, i) => v - xs[i]);

/* Rolling correlation between crowd motion and the beat, searched over lag.
 *
 * Correlating the raw series would mostly measure "both got louder", which is
 * trivially true and says nothing. Correlating the FIRST DIFFERENCES asks the
 * question that matters -- do the pushes line up? -- and the lag search absorbs
 * the delay between a beat leaving the PA and a body answering it.
 */
export function grooveSync(motion, onset, hz, maxLagSeconds = 1.5) {
  if (!motion || !onset || motion.length < hz * 3) return null;
  const dm = diff(motion), don = diff(onset);
  const maxLag = Math.round(maxLagSeconds * hz);
  let best = 0;
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    const a = lag >= 0 ? dm.slice(lag) : dm.slice(0, dm.length + lag);
    const b = lag >= 0 ? don.slice(0, don.length - lag) : don.slice(-lag);
    const r = pearson(a, b);
    if (r != null && r > best) best = r;
  }
  return best <= 0 ? 0 : Math.round(best * 100);
}
