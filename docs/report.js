/* Set report: what the loop learned, once the set is over.
 *
 * A DJ does not want a live meter after the fact -- they want to know which
 * track landed, which genre the room was actually there for, and where the peak
 * was. This renders that from the session's own record: the crowd-energy log
 * sampled at 1 Hz, and the DJ engine's track history with the measured response
 * each track earned.
 *
 * Everything here is derived from measurements already taken. Nothing is
 * recomputed, smoothed or inferred after the fact, and a session too short to
 * support a claim says so rather than showing a confident-looking zero.
 */

const COLD = [34, 211, 238], WARM = [251, 191, 36], HOT = [244, 63, 94];

function ramp(e) {
  if (e == null) return "rgb(120,116,134)";
  const t = Math.max(0, Math.min(1, e / 100));
  const [a, b, k] = t < 0.5 ? [COLD, WARM, t / 0.5] : [WARM, HOT, (t - 0.5) / 0.5];
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * k)).join(",")})`;
}

const fmt = (s) => {
  s = Math.max(0, Math.round(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const esc = (x) => String(x).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function buildReport({ log, dj, tracker, duration, peopleSeen, backend, size }) {
  const scored = dj.history.filter((h) => h.response != null);
  const energies = log.map((p) => p.e).filter((e) => e != null);
  const peak = energies.length ? Math.max(...energies) : null;
  const peakAt = peak == null ? null : log.find((p) => p.e === peak);
  const avg = mean(energies);
  const cmp = dj.loopCompare(1);

  /* ---- headline numbers ------------------------------------------------ */
  const stats = [
    ["Set length", fmt(duration)],
    ["Tracks played", String(dj.history.length)],
    ["Peak energy", peak == null ? "--" : Math.round(peak)],
    ["Average", avg == null ? "--" : Math.round(avg)],
    ["People seen", String(peopleSeen)],
    ["Picks corrected", cmp.compared ? `${cmp.corrected}/${cmp.compared}` : "--"],
  ];

  /* ---- top tracks ------------------------------------------------------- */
  const ranked = [...scored].sort((a, b) => b.response - a.response);
  const best = ranked[0]?.response ?? 1;

  const trackRows = ranked.length
    ? ranked.map((h, i) => `
        <li>
          <span class="r-pos">${String(i + 1).padStart(2, "0")}</span>
          <span class="r-main">
            <span class="r-title">${esc(h.title)} <em>${esc(h.artist)}</em></span>
            <span class="r-meta">${esc(h.genre)} · ${h.bpm} BPM${
              h.openPick ? ` · open loop wanted ${esc(h.openPick.title)}` : ""}</span>
            <span class="r-bar"><i style="width:${Math.max(3, 100 * h.response / best)}%;
              background:${ramp(h.response)}"></i></span>
          </span>
          <span class="r-num" style="color:${ramp(h.response)}">${Math.round(h.response)}</span>
          <span class="r-delta ${h.delta >= 0 ? "up" : "down"}">${
            h.delta == null ? "" : (h.delta >= 0 ? "+" : "") + Math.round(h.delta) + "%"}</span>
        </li>`).join("")
    : `<li class="r-empty">No track collected a response — the set was too short,
         or nobody was in frame long enough to measure one.</li>`;

  /* ---- genres ----------------------------------------------------------- */
  const genres = dj.genreScores();
  const gTop = genres.length ? genres[0].score || 1 : 1;
  const genreRows = genres.length
    ? genres.map((g) => `
        <div class="g-row">
          <div class="g-top"><span class="g-name">${esc(g.genre)}</span>
            <span class="g-score">${Math.round(g.score)}</span></div>
          <div class="g-bar"><div class="g-fill"
            style="width:${Math.max(4, 100 * g.score / gTop)}%"></div></div>
          <div class="g-n">${g.tracks} track${g.tracks === 1 ? "" : "s"}</div>
        </div>`).join("")
    : `<p class="r-empty">Nothing scored yet.</p>`;

  /* ---- the verdict sentence --------------------------------------------- */
  let verdict;
  if (!scored.length) {
    verdict = "Too short to conclude anything — run a longer set.";
  } else if (!cmp.compared) {
    verdict = "The loop never reached a decision point.";
  } else if (cmp.corrected === 0) {
    verdict = `The crowd agreed with the audio-feature match every time
      (${cmp.compared}/${cmp.compared}). On this floor the open loop would have
      done just as well.`;
  } else {
    const pct = Math.round(100 * cmp.corrected / cmp.compared);
    verdict = `The crowd overruled the audio-feature match on <b>${cmp.corrected}
      of ${cmp.compared}</b> decisions (${pct}%). ${
      genres.length ? `This room was here for <b>${esc(genres[0].genre)}</b>.` : ""}`;
  }

  return { stats, trackRows, genreRows, verdict, peakAt, log, duration };
}

/* ---- the full-session energy timeline ---------------------------------- */
export function drawTimeline(canvas, { log, dj, duration, peakAt }) {
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const W = canvas.width = Math.round(canvas.clientWidth * dpr);
  const H = canvas.height = Math.round(canvas.clientHeight * dpr);
  ctx.clearRect(0, 0, W, H);
  if (log.length < 2 || !duration) return;

  const pad = 6 * dpr;
  const x = (t) => (t / duration) * W;
  const y = (e) => H - pad - (Math.max(0, Math.min(100, e)) / 100) * (H - pad * 2);

  // Track boundaries first, so the curve reads on top of them.
  ctx.strokeStyle = "rgba(255,255,255,.07)";
  ctx.lineWidth = dpr;
  for (const h of dj.history) {
    ctx.beginPath();
    ctx.moveTo(x(h.start), pad);
    ctx.lineTo(x(h.start), H - pad);
    ctx.stroke();
  }

  // Energy curve, coloured by its own value so the ramp still means something.
  ctx.lineWidth = 2 * dpr;
  ctx.lineJoin = "round";
  for (let i = 1; i < log.length; i++) {
    ctx.beginPath();
    ctx.moveTo(x(log[i - 1].t), y(log[i - 1].e));
    ctx.lineTo(x(log[i].t), y(log[i].e));
    ctx.strokeStyle = ramp((log[i - 1].e + log[i].e) / 2);
    ctx.stroke();
  }

  // Fill under the curve, very faint.
  ctx.beginPath();
  ctx.moveTo(x(log[0].t), H - pad);
  for (const p of log) ctx.lineTo(x(p.t), y(p.e));
  ctx.lineTo(x(log[log.length - 1].t), H - pad);
  ctx.closePath();
  ctx.fillStyle = "rgba(244,63,94,.07)";
  ctx.fill();

  // The peak: the one moment worth pointing at.
  if (peakAt) {
    ctx.strokeStyle = "rgba(251,191,36,.75)";
    ctx.setLineDash([3 * dpr, 3 * dpr]);
    ctx.beginPath();
    ctx.moveTo(x(peakAt.t), pad);
    ctx.lineTo(x(peakAt.t), H - pad);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(251,191,36,.95)";
    ctx.font = `${9 * dpr}px ui-monospace, monospace`;
    const label = `PEAK ${Math.round(peakAt.e)} @ ${fmt(peakAt.t)}`;
    const tw = ctx.measureText(label).width;
    const lx = Math.min(W - tw - 4 * dpr, Math.max(4 * dpr, x(peakAt.t) + 5 * dpr));
    ctx.fillText(label, lx, pad + 9 * dpr);
  }
}
