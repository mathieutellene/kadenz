/* CrowdPulse dashboard.
   Frames arrive as binary WebSocket messages (4-byte timestamp + JPEG, latest
   wins); everything else arrives as JSON. The colour ramp used for the energy
   meter and the AI view matches engine/annotate.py on purpose. */
const $ = (id) => document.getElementById(id);

let ws = null, chart = null;
let audioMeta = null, videoAR = 16 / 9;
let energyData = [], peopleData = [], levelData = [], eventPoints = [], dropLines = [];
let lastChartUpdate = 0, lastMetricT = 0, prevEnergy = null, lastStart = null;

const fmt = (t) => {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};

/* Cyan -> amber -> hot pink, same ramp as the video overlays. */
function energyColor(e) {
  if (e == null) return "rgb(125,122,140)";
  const t = Math.max(0, Math.min(1, e / 100));
  const stops = t < 0.5
    ? [[34, 211, 238], [251, 191, 36], t / 0.5]
    : [[251, 191, 36], [244, 63, 94], (t - 0.5) / 0.5];
  const [a, b, k] = stops;
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * k)).join(",")})`;
}

function fitFrame() {
  const area = document.querySelector(".stagetop").getBoundingClientRect();
  if (area.width < 2 || area.height < 2) return;
  let w = area.width, h = w / videoAR;
  if (h > area.height) { h = area.height; w = h * videoAR; }
  const wrap = $("frame-wrap");
  wrap.style.width = `${Math.round(w)}px`;
  wrap.style.height = `${Math.round(h)}px`;
}

let statusTimer = null;
function setStatusLine(text) {
  $("statusline").textContent = text || "";
  if (statusTimer) clearTimeout(statusTimer);
  if (text) statusTimer = setTimeout(() => { $("statusline").textContent = ""; }, 12000);
}
function setStatus(text, cls) {
  const el = $("status");
  el.textContent = text;
  el.className = `status ${cls || ""}`;
}

/* ------------------------------------------------------------- AI view PiP */
const AI_EDGES = [[5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6]];

function drawAIView(msg) {
  const cv = $("aiview");
  const sc = Math.min(236 / msg.w, 150 / msg.h);
  const w = Math.round(msg.w * sc), h = Math.round(msg.h * sc);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  cv.style.display = "block";
  const g = cv.getContext("2d");
  g.clearRect(0, 0, w, h);
  g.font = "9px ui-monospace, monospace";
  for (const it of msg.items || []) {
    const col = energyColor(it.e);
    const [x1, y1, x2, y2] = it.b.map((v) => v * sc);
    g.strokeStyle = col;
    g.lineWidth = 1;
    const L = Math.max(3, Math.min(7, (x2 - x1) * 0.2));
    g.beginPath();
    g.moveTo(x1, y1 + L); g.lineTo(x1, y1); g.lineTo(x1 + L, y1);
    g.moveTo(x2 - L, y1); g.lineTo(x2, y1); g.lineTo(x2, y1 + L);
    g.moveTo(x1, y2 - L); g.lineTo(x1, y2); g.lineTo(x1 + L, y2);
    g.moveTo(x2 - L, y2); g.lineTo(x2, y2); g.lineTo(x2, y2 - L);
    g.stroke();
    if (it.k && it.c) {
      g.lineWidth = 1.3;
      g.beginPath();
      for (const [a, b] of AI_EDGES) {
        if (it.c[a] > 0.45 && it.c[b] > 0.45) {
          g.moveTo(it.k[a][0] * sc, it.k[a][1] * sc);
          g.lineTo(it.k[b][0] * sc, it.k[b][1] * sc);
        }
      }
      g.stroke();
      g.fillStyle = "#eceaf3";
      for (let j = 0; j < 17; j++) {
        if (it.c[j] > 0.45) {
          g.beginPath(); g.arc(it.k[j][0] * sc, it.k[j][1] * sc, 1.2, 0, 7); g.fill();
        }
      }
    }
  }
  g.fillStyle = "rgba(125,122,140,.85)";
  g.fillText("MODEL VIEW", 6, 12);
}

/* ------------------------------------------------------------------- chart */
function audioSeriesData(meta) {
  const rate = meta.rate || 10;
  return (meta.rms || []).map((v, i) => [i / rate, Math.round(v * 100)]);
}

function initChart() {
  if (chart) chart.dispose();
  chart = echarts.init($("timeline"), null, { renderer: "canvas" });
  energyData = []; peopleData = []; eventPoints = [];
  levelData = audioMeta?.rms?.length ? audioSeriesData(audioMeta) : [];
  dropLines = (audioMeta?.drops || []).map((d) => ({ xAxis: d }));
  chart.setOption({
    animation: false,
    grid: { left: 38, right: 38, top: 20, bottom: 22 },
    tooltip: {
      trigger: "axis", backgroundColor: "#16161d", borderColor: "#32323f",
      textStyle: { color: "#eceaf3", fontSize: 10 },
    },
    legend: {
      data: ["Level", "Energy", "People"], top: 0, right: 8,
      textStyle: { color: "#7d7a8c", fontSize: 9 },
      icon: "roundRect", itemWidth: 10, itemHeight: 2,
    },
    xAxis: {
      type: "value", min: 0, max: 15,
      axisLabel: { color: "#4e4b5c", fontSize: 9, formatter: (v) => fmt(v) },
      splitLine: { lineStyle: { color: "#16161d" } },
      axisLine: { lineStyle: { color: "#23232e" } },
    },
    yAxis: [
      { type: "value", min: 0, max: 100,
        axisLabel: { color: "#4e4b5c", fontSize: 9 },
        splitLine: { lineStyle: { color: "#16161d" } } },
      { type: "value", min: 0,
        axisLabel: { color: "#22d3ee", fontSize: 9 }, splitLine: { show: false } },
    ],
    series: [
      { id: "level", name: "Level", type: "line", color: "#4e4b5c", data: levelData,
        showSymbol: false, lineStyle: { width: 1, opacity: .6 },
        areaStyle: { color: "#7d7a8c", opacity: .10 }, silent: true, z: 1 },
      { id: "energy", name: "Energy", type: "line", color: "#f43f5e", data: energyData,
        showSymbol: false, smooth: 0.25, lineStyle: { width: 2 },
        areaStyle: { color: "#f43f5e", opacity: .10 }, z: 3,
        markLine: { silent: true, symbol: "none", label: { show: false },
          lineStyle: { color: "#fbbf24", type: "dashed", width: 1 }, data: dropLines },
        markPoint: { data: eventPoints, symbolSize: 22, label: { fontSize: 9 } } },
      { id: "people", name: "People", type: "line", color: "#22d3ee", yAxisIndex: 1,
        data: peopleData, showSymbol: false, lineStyle: { width: 1.2 }, z: 2 },
    ],
  });
}

function refreshChart(force) {
  const now = Date.now();
  if (!force && now - lastChartUpdate < 900) return;
  lastChartUpdate = now;
  chart.setOption({
    xAxis: { min: 0, max: Math.max(15, Math.ceil(lastMetricT + 1)) },
    series: [
      { id: "level", data: levelData },
      { id: "energy", data: energyData, markPoint: { data: eventPoints }, markLine: { data: dropLines } },
      { id: "people", data: peopleData },
    ],
  });
}

/* --------------------------------------------------------------- DJ panels */
function renderDJ(msg) {
  const now = msg.now;
  if (now) {
    $("np-title").textContent = `${now.title} — ${now.artist}`;
    $("np-meta").textContent =
      `${now.genre.toUpperCase()} · ${now.bpm} BPM · ${now.key} · ${fmt(now.elapsed)}`;
    const r = now.response;
    $("np-resp").textContent = r == null ? "--" : Math.round(r);
    $("np-resp").style.color = r == null ? "" : energyColor(r);
    $("resp-fill").style.width = `${Math.max(0, Math.min(100, r || 0))}%`;
  }
  const list = $("next-list");
  const next = msg.next || [];
  list.innerHTML = next.length ? next.map((t, i) => `
    <div class="next-item${i ? " alt" : ""}">
      <div class="next-top">
        <span class="next-title">${t.title} — ${t.artist}</span>
        <span class="next-bpm">${t.bpm} BPM${t.mixable ? "" : " ⚠"}</span>
      </div>
      <div class="next-why">${t.reason}</div>
    </div>`).join("") : '<div class="empty">Listening to how the crowd reacts...</div>';

  const gl = $("genre-list");
  const gs = msg.genres || [];
  const top = gs.length ? gs[0].score || 1 : 1;
  gl.innerHTML = gs.length ? gs.map((g) => `
    <div class="g-row">
      <div class="g-top"><span class="g-name">${g.genre}</span>
        <span class="g-score">${Math.round(g.score)}</span></div>
      <div class="g-bar"><div class="g-fill" style="width:${Math.max(4, 100 * g.score / top)}%"></div></div>
    </div>`).join("") : '<div class="empty">No tracks scored yet</div>';

  if (msg.changed && msg.last && msg.last.delta != null) {
    const d = msg.last.delta;
    setStatusLine(`${d >= 0 ? "▲" : "▼"} "${msg.last.title}" (${msg.last.genre}) ${d >= 0 ? "+" : ""}${Math.round(d)}% vs set average`);
  }
}

/* ---------------------------------------------------------------- protocol */
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => setStatus("READY");
  ws.onclose = () => { setStatus("RECONNECTING", "warn"); setTimeout(connect, 1500); };
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) handleFrame(e.data);
    else handle(JSON.parse(e.data));
  };
}
const send = (o) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };

let lastFrameUrl = null;
function handleFrame(buf) {
  const t = new DataView(buf).getUint32(0) / 1000;
  const url = URL.createObjectURL(new Blob([buf.slice(4)], { type: "image/jpeg" }));
  const img = $("frame");
  img.onload = () => { if (lastFrameUrl) URL.revokeObjectURL(lastFrameUrl); lastFrameUrl = url; };
  img.src = url;
  img.style.display = "block";
  $("placeholder").style.display = "none";
  $("tclock").style.display = "block";
  $("tclock").textContent = fmt(t);
}

function handle(msg) {
  switch (msg.type) {
    case "hello": {
      audioMeta = msg.audio;
      const v = msg.video;
      if (v.width && v.height) { videoAR = v.width / v.height; fitFrame(); }
      initChart();
      setStatusLine(`${v.name} · ${v.width}×${v.height} @ ${v.fps}fps${v.duration ? " · " + fmt(v.duration) : ""}`);
      setStatus("LIVE", "live");
      break;
    }
    case "status": setStatusLine(msg.text); break;
    case "audio":
      audioMeta = msg;
      $("v-bpm").textContent = Math.round(msg.bpm);
      levelData = audioSeriesData(msg);
      dropLines = (msg.drops || []).map((d) => ({ xAxis: d }));
      refreshChart(true);
      break;
    case "audio_live":
      if (msg.bpm) $("v-bpm").textContent = Math.round(msg.bpm);
      break;
    case "tracks": drawAIView(msg); break;
    case "dj": renderDJ(msg); break;
    case "metrics": {
      const e = msg.energy;
      $("v-energy").textContent = Math.round(e);
      $("v-energy").style.color = energyColor(e);
      $("energy-fill").style.width = `${Math.max(0, Math.min(100, e))}%`;
      if (prevEnergy != null) {
        const d = e - prevEnergy;
        $("v-trend").textContent = Math.abs(d) < 1 ? "—" : (d > 0 ? `▲ ${d.toFixed(0)}` : `▼ ${Math.abs(d).toFixed(0)}`);
      }
      prevEnergy = e;
      $("v-people").textContent = msg.occupancy;
      $("v-moving").textContent = `${Math.round(msg.participation)}%`;
      $("v-db").textContent = msg.db == null ? "--" : Math.round(msg.db);
      if (msg.bpm) $("v-bpm").textContent = Math.round(msg.bpm);
      const simCells = msg.sim_audio === true;
      $("v-bpm").style.color = simCells ? "var(--warm)" : "";
      $("v-db").style.color = simCells ? "var(--warm)" : "";
      if (simCells && !$("v-bpm").dataset.simmed) {
        $("v-bpm").dataset.simmed = "1";
        $("v-bpm").parentElement.querySelector(".cl").textContent = "BPM · sim";
        $("v-db").parentElement.querySelector(".cl").textContent = "Level dB SPL · sim";
      }
      // Groove sync needs a real beat to correlate against; say so instead of faking it
      if (msg.groove == null) {
        $("v-groove").textContent = msg.sim_audio ? "n/a" : "--";
        $("v-groove").style.color = "var(--dim)";
        $("v-groove").parentElement.title = msg.sim_audio
          ? "Needs real audio: this clip has no music track" : "";
      } else {
        $("v-groove").textContent = `${Math.round(msg.groove)}%`;
        $("v-groove").style.color = "";
      }
      const flux = (msg.in_per_min || 0) - (msg.out_per_min || 0);
      $("v-flux").textContent = `${flux > 0 ? "+" : ""}${flux}`;
      if (msg.proc_fps) $("proc-fps").textContent = `engine ${msg.proc_fps} fps`;
      lastMetricT = msg.t;
      energyData.push([msg.t, e]);
      peopleData.push([msg.t, msg.occupancy]);
      if (msg.db != null) levelData.push([msg.t, Math.max(0, Math.min(100, (msg.db + 60) * 1.67))]);
      refreshChart(false);
      break;
    }
    case "event":
      break;
    case "end": {
      const s = msg.summary || {};
      setStatus("DONE");
      setEventButton(false);
      if (s.duration != null) {
        setStatusLine(`Session complete · avg energy ${s.avg_energy} · peak crowd ${s.max_occupancy} · top moments ${(s.top_moments || []).slice(0, 3).map((m) => fmt(m.t)).join(", ")}`);
      }
      refreshChart(true);
      break;
    }
    case "error": setStatus("ERROR", "warn"); setStatusLine(msg.text); break;
  }
}

/* ----------------------------------------------------------------- actions */
function startAnalysis(cmd) {
  lastStart = { ...cmd };
  audioMeta = null; lastMetricT = 0; prevEnergy = null;
  $("aiview").style.display = "none";
  $("np-title").textContent = "--";
  $("np-meta").textContent = "waiting for the set to start";
  $("np-resp").textContent = "--";
  $("resp-fill").style.width = "0%";
  $("next-list").innerHTML = '<div class="empty">Listening to how the crowd reacts...</div>';
  $("genre-list").innerHTML = '<div class="empty">No tracks scored yet</div>';
  $("v-trend").textContent = "";
  delete $("v-bpm").dataset.simmed;
  $("v-bpm").parentElement.querySelector(".cl").textContent = "BPM";
  $("v-db").parentElement.querySelector(".cl").textContent = "Level dB SPL";
  $("v-bpm").style.color = ""; $("v-db").style.color = "";
  setStatus("STARTING", "warn");
  send(cmd);
}

let eventRunning = false;
function setEventButton(running) {
  eventRunning = running;
  const b = $("btn-event");
  b.textContent = running ? "END EVENT" : "START EVENT";
  b.classList.toggle("primary", !running);
  b.classList.toggle("danger", running);
}
$("btn-event").onclick = () => {
  if (eventRunning) { send({ type: "stop" }); setEventButton(false); return; }
  $("file-input").click();     // pick the floor footage to read
};
$("file-input").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  setStatus("UPLOADING", "warn");
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/upload", { method: "POST", body: form });
  const data = await res.json();
  if (data.ok) { startAnalysis({ type: "start", file: data.name }); setEventButton(true); }
  else { setStatus("ERROR", "warn"); setStatusLine(data.error || "Upload failed"); }
  e.target.value = "";
};

window.addEventListener("resize", () => { if (chart) chart.resize(); fitFrame(); });

initChart();
fitFrame();
connect();
