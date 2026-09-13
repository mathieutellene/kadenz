/* Kadenz live — the whole loop, in a tab.
 *
 * Nothing leaves the device: the camera and microphone streams are read,
 * measured and thrown away frame by frame. There is no server in this build at
 * all, which is also why it can be handed to anyone with a link.
 *
 * The desktop app decouples display from analysis across threads so the video
 * never waits for the model. Same idea here, one thread: render() runs every
 * animation frame off the <video> element and the last known tracks, while
 * analyse() runs its own async loop at whatever rate inference manages. The
 * video is never blocked on a forward pass.
 */
import { DJEngine } from "./dj.js?v=23";
import {
  EDGES, MODELS, Motion, Normaliser, Tracker,
  loadModel, makeLetterbox, postprocess, preprocess,
} from "./vision.js?v=23";
import { AudioAnalyser, grooveSync } from "./audio.js?v=23";
import { Backdrop } from "./backdrop.js?v=23";
import { buildReport, drawTimeline } from "./report.js?v=23";

const TRACK_SECONDS = 14;        // matches config.yaml dj.track_seconds
const METRIC_HZ = 4;
const KP_CONF = 0.45;

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/* Colour ramp, identical to engine/annotate.py: cyan -> amber -> hot pink. */
const COLD = [34, 211, 238], WARM = [251, 191, 36], HOT = [244, 63, 94];
function energyColor(e) {
  if (e == null) return "rgb(120,116,134)";
  const t = clamp(e / 100, 0, 1);
  const [a, b, u] = t < 0.5 ? [COLD, WARM, t / 0.5] : [WARM, HOT, (t - 0.5) / 0.5];
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * u)).join(",")})`;
}

const state = {
  running: false, model: null, backend: "", video: null, stream: null,
  tracker: null, motion: null, norm: null, perNorm: null, dj: null, audio: null,
  tracks: [], energy: null, prevEnergy: null, t0: 0, lastMetric: 0, lastDJ: 0,
  procFps: 0, history: [], djCmp: null, hasAudio: false, db: null, bpm: null,
  groove: null, fatal: null, metricTimer: null, size: 320, swapping: false,
  // The live chart keeps a 20 s window; the report needs the whole set, so
  // this logs at 1 Hz alongside it. 1 Hz for an hour is 3600 points -- nothing.
  sessionLog: [], lastLog: -99, peopleSeen: 0, repaintReport: null,
};

/* ------------------------------------------------------------------ startup */
async function boot() {
  setStatus("REQUESTING CAMERA", "warn");
  let stream;
  try {
    // Audio is best-effort: without it Groove Sync and BPM simply stay blank,
    // which is preferable to blocking the whole demo on a mic permission.
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (e) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 } } });
    } catch (e2) {
      return fail(`Camera permission denied — ${e2.message}. `
        + "Nothing works without it, and nothing is uploaded if you allow it.");
    }
  }
  state.stream = stream;
  state.hasAudio = stream.getAudioTracks().length > 0;

  const video = $("cam");
  video.srcObject = stream;
  await video.play();
  state.video = video;

  try {
    setStatus("LOADING MODEL", "warn");
    const { session, backend, size } = await loadModel(state.size, (s) => setStatusLine(s));
    state.model = session;
    state.backend = backend;
    state.size = size;
  } catch (e) {
    return fail(`Could not load the pose model: ${e.message}`);
  }

  if (state.hasAudio) {
    state.audio = new AudioAnalyser();
    await state.audio.start(stream).catch(() => { state.hasAudio = false; });
  }
  $("audio-note").textContent = state.hasAudio
    ? "mic on — BPM and groove sync are measured"
    : "no mic — BPM and groove sync unavailable";

  state.tracker = new Tracker();
  state.motion = new Motion();
  state.norm = new Normaliser();
  state.perNorm = new Normaliser(4000);
  state.dj = new DJEngine();
  state.sessionLog = [];
  state.lastLog = -99;
  state.peopleSeen = 0;
  state.t0 = performance.now() / 1000;
  state.running = true;

  // The backdrop is decoration; inference is not. Stop it before the first
  // forward pass rather than leaving it competing for the same frame budget.
  backdrop?.stop();
  document.body.classList.add("running");
  $("intro").hidden = true;
  $("stage").hidden = false;
  setStatus("LIVE", "live");
  showBackend();

  // Three independent clocks, same reasoning as the desktop build: drawing must
  // never gate measurement. requestAnimationFrame is throttled to zero in a
  // background tab and in some embedded views, so the metric and DJ loops run
  // on their own timer -- the numbers stay live even when nothing is painting.
  requestAnimationFrame(render);
  state.metricTimer = setInterval(() => {
    if (state.running) tickMetrics(performance.now() / 1000);
  }, 1000 / METRIC_HZ);
  analyse();
}

function fail(msg) {
  state.fatal = msg;
  setStatus("ERROR", "warn");
  $("intro").hidden = false;
  $("intro-error").textContent = msg;
  $("intro-error").hidden = false;
  $("btn-start").disabled = false;
  $("btn-start").textContent = "TRY AGAIN";
}

/* ------------------------------------------------------- analysis (async) */
const infCanvas = document.createElement("canvas");
const infCtx = infCanvas.getContext("2d", { willReadFrequently: true });
let inputBuf = null;

function sizeInputTo(size) {
  if (infCanvas.width === size) return;
  infCanvas.width = infCanvas.height = size;
  inputBuf = new Float32Array(3 * size * size);
}

async function analyse() {
  const ort = window.ort;
  let lb = null;
  while (state.running) {
    const v = state.video;
    if (!v.videoWidth) { await sleep(80); continue; }
    const started = performance.now();
    try {
      const size = state.size;
      sizeInputTo(size);
      if (!lb || lb.srcW !== v.videoWidth || lb.size !== size) {
        lb = { ...makeLetterbox(v.videoWidth, v.videoHeight, size), srcW: v.videoWidth };
      }
      infCtx.fillStyle = "#000";
      infCtx.fillRect(0, 0, size, size);
      infCtx.drawImage(v, lb.dx, lb.dy, lb.w, lb.h);
      preprocess(infCtx, size, inputBuf);

      const feeds = { images: new ort.Tensor("float32", inputBuf, [1, 3, size, size]) };
      const out = await state.model.run(feeds);
      const dets = postprocess(out[Object.keys(out)[0]], lb, v.videoWidth, v.videoHeight);

      const diff = state.motion.ingest(v);
      const live = state.tracker.update(dets);
      for (const t of live) {
        t.raw = state.motion.inBox(diff, t, v.videoWidth, v.videoHeight);
        state.perNorm.add(t.raw);
        t.e = state.perNorm.rank(t.raw);
      }
      state.tracks = live;

      const crowd = live.length ? live.reduce((s, t) => s + t.raw, 0) / live.length : 0;
      state.norm.add(crowd);
      const ranked = state.norm.rank(crowd);
      if (ranked != null) {
        // Same EMA the desktop metrics use: the raw percentile is far too jumpy
        // to drive a meter a human is reading.
        state.energy = state.energy == null ? ranked : state.energy * 0.72 + ranked * 0.28;
      }
      const ms = performance.now() - started;
      state.procFps = state.procFps ? state.procFps * 0.8 + (1000 / ms) * 0.2 : 1000 / ms;
    } catch (e) {
      console.error("[kadenz] analysis:", e);
      await sleep(400);
    }
    await sleep(0);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------------------- render loop */
function render() {
  if (!state.running) return;
  const v = state.video, cv = $("overlay"), ctx = cv.getContext("2d");
  if (v.videoWidth && (cv.width !== v.videoWidth || cv.height !== v.videoHeight)) {
    cv.width = v.videoWidth; cv.height = v.videoHeight;
  }
  ctx.clearRect(0, 0, cv.width, cv.height);

  // Same overlay rules as engine/annotate.py: hairline strokes, corner ticks
  // instead of boxes, a limb only when BOTH joints are confident.
  const s = Math.max(cv.width, cv.height) / 960;
  ctx.lineCap = "round";
  for (const t of state.tracks) {
    const col = energyColor(t.e);
    ctx.strokeStyle = col;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = Math.max(1, 1.5 * s);
    ctx.beginPath();
    for (const [a, b] of EDGES) {
      if (t.kconf[a] > KP_CONF && t.kconf[b] > KP_CONF) {
        ctx.moveTo(t.kpts[a * 2], t.kpts[a * 2 + 1]);
        ctx.lineTo(t.kpts[b * 2], t.kpts[b * 2 + 1]);
      }
    }
    ctx.stroke();
    cornerTicks(ctx, t, s);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "rgba(245,242,252,.9)";
    for (let k = 0; k < 17; k++) {
      if (t.kconf[k] > KP_CONF) {
        ctx.beginPath();
        ctx.arc(t.kpts[k * 2], t.kpts[k * 2 + 1], Math.max(1, 1.8 * s), 0, 6.284);
        ctx.fill();
      }
    }
  }
  ctx.globalAlpha = 1;

  drawChart();
  requestAnimationFrame(render);
}

function cornerTicks(ctx, t, s) {
  const len = Math.max(4, Math.min((t.x2 - t.x1) * 0.18, 13 * s));
  ctx.lineWidth = Math.max(1, 1.4 * s);
  ctx.beginPath();
  for (const [cx, cy, dx, dy] of [
    [t.x1, t.y1, 1, 1], [t.x2, t.y1, -1, 1], [t.x1, t.y2, 1, -1], [t.x2, t.y2, -1, -1],
  ]) {
    ctx.moveTo(cx, cy); ctx.lineTo(cx + dx * len, cy);
    ctx.moveTo(cx, cy); ctx.lineTo(cx, cy + dy * len);
  }
  ctx.stroke();
}

/* --------------------------------------------------------------- metrics */
function tickMetrics(now) {
  const t = now - state.t0;
  const e = state.energy;
  $("v-energy").textContent = e == null ? "--" : Math.round(e);
  $("v-energy").style.color = e == null ? "" : energyColor(e);
  $("energy-fill").style.width = `${clamp(e ?? 0, 0, 100)}%`;
  if (e != null && state.prevEnergy != null) {
    const d = e - state.prevEnergy;
    $("v-trend").textContent = Math.abs(d) < 1 ? "—" : d > 0 ? `▲ ${d.toFixed(0)}` : `▼ ${Math.abs(d).toFixed(0)}`;
  }
  state.prevEnergy = e;

  const people = state.tracks.length;
  $("v-people").textContent = people;
  const moving = state.tracks.filter((x) => (x.e ?? 0) > 25).length;
  $("v-moving").textContent = people ? `${Math.round((100 * moving) / people)}%` : "--";

  const a = state.audio;
  const audioTick = a?.ready ? a.tick(t) : null;
  if (audioTick) { state.db = audioTick.db; state.bpm = a.bpm(); }
  $("v-db").textContent = state.db == null ? "--" : Math.round(state.db);
  $("v-bpm").textContent = state.bpm == null ? "--" : state.bpm;

  state.history.push({ t, e: e ?? 0 });
  if (t - state.lastLog >= 1) {
    state.lastLog = t;
    state.sessionLog.push({ t, e: e ?? 0, people });
  }
  state.peopleSeen = Math.max(state.peopleSeen, state.tracker?.entries ?? 0);
  const cut = t - 20;
  while (state.history.length && state.history[0].t < cut) state.history.shift();

  if (a?.ready && state.history.length > METRIC_HZ * 4) {
    const grid = state.history.map((h) => h.t);
    const env = a.envelopeOn(grid);
    state.groove = grooveSync(state.history.map((h) => h.e), env, METRIC_HZ);
  }
  $("v-groove").textContent = state.groove == null ? "--" : `${state.groove}%`;

  const mins = Math.max(t / 60, 1 / 60);
  const flux = Math.round((state.tracker.entries - state.tracker.exits) / mins);
  $("v-flux").textContent = `${flux > 0 ? "+" : ""}${flux}`;
  $("proc-fps").textContent = `engine ${state.procFps.toFixed(1)} fps`;
  $("clock").textContent = fmt(t);

  djTick(t);
}

function fmt(s) { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }

/* --------------------------------------------------------------- the loop */
function djTick(t) {
  const dj = state.dj;
  // With nobody in frame there is no crowd response to measure, and feeding the
  // recommender a zero would be exactly the failure the desktop build refuses
  // for silent audio: a fabricated reward dressed up as a measurement. Hold the
  // set instead -- the loop resumes the moment someone is detected again.
  if (!state.tracks.length) {
    $("empty-hint").hidden = false;
    return;
  }
  $("empty-hint").hidden = true;
  if (state.energy != null) dj.sample(state.energy);
  const cur = dj.current;
  if (!cur || t - cur.start >= TRACK_SECONDS) {
    if (cur) dj.closeCurrent(t);
    const cmp = dj.loopCompare(3, true);   // count divergence at decision points
    if (cmp.closed.length) {
      const op = cmp.open[0] ?? null;
      dj.startTrack(cmp.closed[0], t, op && { title: op.title, genre: op.genre });
    }
    renderDJ(dj.loopCompare(3), true);
    state.lastDJ = t;
  } else if (t - state.lastDJ >= 2) {
    renderDJ(dj.loopCompare(3), false);
    state.lastDJ = t;
  }
}

function trackLine(tr) {
  return `${tr.title} — ${tr.artist} <span class="bpm">${tr.bpm} BPM${tr.mixable === false ? " ⚠" : ""}</span>`;
}

function renderDJ(cmp, changed) {
  const dj = state.dj, cur = dj.current;
  if (cur) {
    $("np-title").textContent = `${cur.title} — ${cur.artist}`;
    const el = (performance.now() / 1000 - state.t0) - cur.start;
    $("np-meta").textContent =
      `${cur.genre.toUpperCase()} · ${cur.bpm} BPM · ${cur.key} · ${fmt(el)}`;
    const r = cur.samples.length ? cur.samples.reduce((s, x) => s + x, 0) / cur.samples.length : null;
    $("np-resp").textContent = r == null ? "--" : Math.round(r);
    $("np-resp").style.color = r == null ? "" : energyColor(r);
    $("resp-fill").style.width = `${clamp(r ?? 0, 0, 100)}%`;
    for (const k of ["energy", "dance", "valence"]) {
      $(`f-${k}`).style.width = `${Math.round(cur[k] * 100)}%`;
      $(`fv-${k}`).textContent = cur[k].toFixed(2).slice(1);
    }
  }
  const o = cmp.open[0];
  if (o) { $("open-title").innerHTML = trackLine(o); $("open-why").textContent = o.reason; }
  const c = cmp.closed[0];
  if (c) {
    $("next-title").innerHTML = trackLine(c);
    $("next-why").textContent = c.reason;
    $("next-alts").innerHTML = cmp.closed.slice(1, 3).map((x) =>
      `<div class="alt"><div class="at">${x.title}</div>
       <div class="ab">${x.genre.toUpperCase()} · ${x.bpm}</div></div>`).join("");
  }
  const vd = $("loop-verdict");
  vd.className = cmp.agree === false ? "verdict diverge" : cmp.agree ? "verdict agree" : "verdict";
  $("verdict-text").textContent = cmp.note ?? "the floor has not spoken yet";
  $("loop-rate").textContent = cmp.compared ? `set so far: ${cmp.corrected}/${cmp.compared} corrected` : "";

  const gs = dj.genreScores();
  const top = gs.length ? gs[0].score || 1 : 1;
  $("genre-list").innerHTML = gs.length
    ? gs.map((g) => `<div class="g-row">
        <div class="g-top"><span class="g-name">${g.genre}</span>
        <span class="g-score">${Math.round(g.score)}</span></div>
        <div class="g-bar"><div class="g-fill" style="width:${Math.max(4, (100 * g.score) / top)}%"></div></div>
      </div>`).join("")
    : '<div class="empty">No tracks scored yet</div>';

  if (changed) {
    const last = dj.history[dj.history.length - 1];
    if (last?.delta != null) {
      setStatusLine(`${last.delta >= 0 ? "▲" : "▼"} "${last.title}" (${last.genre}) `
        + `${last.delta >= 0 ? "+" : ""}${Math.round(last.delta)}% vs set average`);
    }
  }
}

/* ----------------------------------------------------------------- chart */
function drawChart() {
  const cv = $("chart"), ctx = cv.getContext("2d");
  const w = cv.width = cv.clientWidth * devicePixelRatio;
  const h = cv.height = cv.clientHeight * devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  const pts = state.history;
  if (pts.length < 2) return;
  const t0 = pts[0].t, span = Math.max(pts[pts.length - 1].t - t0, 1);
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = ((p.t - t0) / span) * w;
    const y = h - (clamp(p.e, 0, 100) / 100) * (h - 4) - 2;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.strokeStyle = energyColor(state.energy);
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.stroke();
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.fillStyle = "rgba(244,63,94,.10)";
  ctx.fill();
}

/* ------------------------------------------------------------------- misc */
/* The same quality/speed dial as `det_imgsz` on the desktop. The bigger graph is
   a separate 13 MB download, so it is only fetched when someone asks for it. */
async function setSize(size) {
  if (state.swapping || size === state.size) return;
  state.swapping = true;
  $("btn-size").disabled = true;
  setStatusLine(`loading ${MODELS[size].label} model (${size}px)…`);
  try {
    const r = await loadModel(size, (s) => setStatusLine(s));
    state.model = r.session;
    state.backend = r.backend;
    state.size = r.size;
    showBackend();
  } catch (e) {
    setStatusLine(`could not load the ${size}px model: ${e.message}`);
  } finally {
    state.swapping = false;
    $("btn-size").disabled = false;
  }
}

function showBackend() {
  const other = state.size === 320 ? 640 : 320;
  $("btn-size").textContent = `${state.size}px · switch to ${other}`;
  setStatusLine(`${state.size}px on ${state.backend.toUpperCase()} · nothing leaves this device`);
}

function setStatus(text, cls = "") { const el = $("status"); el.textContent = text; el.className = `status ${cls}`; }
function setStatusLine(text) { $("statusline").textContent = text; }

function stop() {
  const duration = state.running ? performance.now() / 1000 - state.t0 : 0;
  state.running = false;
  clearInterval(state.metricTimer);
  state.audio?.stop();
  state.stream?.getTracks().forEach((t) => t.stop());
  setStatus("STOPPED");
  $("btn-stop").hidden = true;
  $("btn-size").hidden = true;
  document.body.classList.remove("running");
  $("stage").hidden = true;
  showReport(duration);
}

/* The set is over: say what it learned, rather than dumping the user back on a
   landing page as if nothing had happened. */
function showReport(duration) {
  const r = buildReport({
    log: state.sessionLog, dj: state.dj, tracker: state.tracker,
    duration, peopleSeen: state.peopleSeen,
    backend: state.backend, size: state.size,
  });
  $("rep-stats").innerHTML = r.stats.map(([k, v]) =>
    `<div class="rs"><div class="rs-v">${v}</div><div class="rs-k">${k}</div></div>`).join("");
  $("rep-tracks").innerHTML = r.trackRows;
  $("rep-genres").innerHTML = r.genreRows;
  $("rep-verdict").innerHTML = r.verdict;
  $("report").hidden = false;
  // Draw whenever the canvas actually has a size, which is not necessarily now:
  // a report opened in a background tab, a collapsed pane or a window that has
  // not laid out yet all report zero width, and a one-shot draw there leaves a
  // permanently blank chart. A ResizeObserver covers the initial layout and
  // every later resize with the same three lines.
  const canvas = $("rep-timeline");
  state.repaintReport?.disconnect();
  const paint = () => {
    if (canvas.clientWidth > 0) {
      drawTimeline(canvas, {
        log: r.log, dj: state.dj, duration: r.duration, peakAt: r.peakAt,
      });
    }
  };
  const ro = new ResizeObserver(paint);
  ro.observe(canvas);
  state.repaintReport = ro;
  paint();
}

function closeReport() {
  state.repaintReport?.disconnect();
  state.repaintReport = null;
  $("report").hidden = true;
  $("intro").hidden = false;
  backdrop?.start();
  $("btn-start").disabled = false;
  $("btn-start").textContent = "Run another set";
}

$("btn-start").addEventListener("click", () => {
  $("btn-start").disabled = true;
  $("intro-error").hidden = true;
  $("btn-stop").hidden = false;
  $("btn-size").hidden = false;
  boot();
});
$("btn-stop").addEventListener("click", stop);
$("btn-size").addEventListener("click", () => setSize(state.size === 320 ? 640 : 320));
$("rep-close").addEventListener("click", closeReport);

const backdrop = new Backdrop($("bg"));
backdrop.start();

if (!navigator.mediaDevices?.getUserMedia) {
  fail("This browser has no camera API. Chrome, Edge, Firefox or Safari on a desktop all work.");
}
