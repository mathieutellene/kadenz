/* Closed-loop DJ intelligence, browser edition.
 *
 * A port of engine/dj.py. The scoring, the mixability cliff and the divergence
 * bookkeeping are identical, because the point of this page is that a visitor
 * can reproduce the README's claim on their own kitchen floor. The catalogue
 * itself is NOT duplicated here -- it comes from catalogue.js, which is
 * generated from the Python (see scripts/build_catalogue.py) so the two can
 * never drift.
 *
 * The one thing that is NOT bit-identical is the exploration draw; see the
 * note on mulberry32 below.
 */
import { CATALOGUE, MIXABLE_BPM_PCT, SIM_W, BPM_LO, BPM_HI } from "./catalogue.js";

const tempoNorm = (bpm) => (bpm - BPM_LO) / (BPM_HI - BPM_LO);

export function featureDistance(a, b) {
  const d =
    SIM_W.energy * Math.abs(a.energy - b.energy) +
    SIM_W.dance * Math.abs(a.dance - b.dance) +
    SIM_W.valence * Math.abs((a.valence ?? 0.5) - (b.valence ?? 0.5)) +
    SIM_W.tempo * Math.abs(tempoNorm(a.bpm) - tempoNorm(b.bpm));
  return Math.min(1, d);
}

const mean = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
const pct = (x) => `${Math.round(x * 100)}%`;

/* Deterministic PRNG so a reload replays the same exploration sequence. Note it
   is NOT the same sequence Python draws: mulberry32 and Mersenne Twister cannot
   be made to agree, so this page explores the catalogue in a different order
   than scripts/sim_loop.py. It reaches the same conclusions — same divergence
   count, same learned genre table — by a different route, which is what a
   bandit is supposed to do. */
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class DJEngine {
  constructor(seed = 7) {
    this.rng = mulberry32(seed);
    this.history = [];
    this.genreReward = new Map();
    this.tempoReward = new Map();
    this.current = null;
    this.played = new Set();
    this.divergences = 0;
    this.comparisons = 0;
  }

  static band(bpm) { return Math.floor(bpm / 5) * 5; }

  _reward(map, key) { return map.get(key) || []; }

  _baseline() {
    const scored = this.history.filter((h) => h.response != null).map((h) => h.response);
    return scored.length ? mean(scored) : null;
  }

  _candidates() {
    return CATALOGUE.filter((t) => !(this.current && t.title === this.current.title));
  }

  _mixable(tr) {
    const cur = this.current?.bpm;
    if (!cur) return [true, 0];
    const drift = Math.abs(tr.bpm - cur) / cur;
    return [drift <= MIXABLE_BPM_PCT, drift];
  }

  startTrack(track, t, openPick = null) {
    this.current = { ...track, start: t, samples: [], openPick };
    this.played.add(track.title);
    return this.current;
  }

  /* Crowd energy measured by the vision pipeline. This is the reward. */
  sample(energy) {
    if (this.current) this.current.samples.push(energy);
  }

  closeCurrent(t) {
    const cur = this.current;
    if (!cur) return null;
    const response = cur.samples.length ? mean(cur.samples) : null;
    const base = this._baseline();
    const delta =
      response == null || base == null || base <= 0 ? null : (100 * (response - base)) / base;
    const entry = {
      title: cur.title, artist: cur.artist, genre: cur.genre, bpm: cur.bpm,
      key: cur.key, energy: cur.energy, dance: cur.dance, valence: cur.valence,
      response, delta, start: cur.start, end: t, openPick: cur.openPick,
    };
    this.history.push(entry);
    if (response != null) {
      const g = this._reward(this.genreReward, entry.genre);
      g.push(response); this.genreReward.set(entry.genre, g);
      const b = this._reward(this.tempoReward, DJEngine.band(entry.bpm));
      b.push(response); this.tempoReward.set(DJEngine.band(entry.bpm), b);
    }
    this.current = null;
    return entry;
  }

  /* ---- OPEN LOOP: content only, never sees the room -------------------- */
  recommendOpen(n = 3) {
    const cur = this.current;
    const out = this._candidates().map((tr) => {
      let score;
      if (cur) {
        const sim = 1 - featureDistance(cur, tr);
        score = 100 * (0.7 * sim + 0.3 * tr.pop);
      } else {
        score = 100 * tr.pop;
      }
      if (this.played.has(tr.title)) score -= 25;
      const [mixable] = this._mixable(tr);
      return { ...tr, score: +score.toFixed(1), mixable, reason: this._reasonOpen(tr, cur) };
    });
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, n);
  }

  _reasonOpen(tr, cur) {
    if (!cur) return `Highest popularity in catalogue (${pct(tr.pop)})`;
    const sim = 1 - featureDistance(cur, tr);
    return `${pct(sim)} audio-feature match, popularity ${pct(tr.pop)}`;
  }

  /* ---- CLOSED LOOP: the same catalogue, corrected by the crowd ---------- */
  recommend(n = 3) {
    const all = [...this.genreReward.values()].flat();
    const globalMean = mean(all);
    const out = this._candidates().map((tr) => {
      const g = this._reward(this.genreReward, tr.genre);
      const b = this._reward(this.tempoReward, DJEngine.band(tr.bpm));
      const gScore = g.length ? mean(g) : globalMean;
      const bScore = b.length ? mean(b) : globalMean;
      let score = 0.55 * gScore + 0.3 * bScore + 15 * tr.dance;
      const [mixable, drift] = this._mixable(tr);
      // A track the DJ cannot beatmatch is unusable however well it would land.
      if (!mixable) score -= 30 + 300 * (drift - MIXABLE_BPM_PCT);
      if (this.played.has(tr.title)) score -= 25;
      score += this.rng() * 4;   // exploration: keep sampling the room
      return { ...tr, score: +score.toFixed(1), mixable, reason: this._reason(tr, g, b) };
    });
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, n);
  }

  _reason(tr, g, b) {
    if (g.length) {
      const avg = mean(g);
      const base = this._baseline();
      if (base && base > 0) {
        const d = (100 * (avg - base)) / base;
        if (d >= 5) return `${tr.genre} running +${Math.round(d)}% above tonight's average`;
        if (d <= -5) return `${tr.genre} under-performed ${Math.round(d)}% - risky pick`;
      }
      return `${tr.genre} matches tonight's profile`;
    }
    if (b.length) {
      const lo = DJEngine.band(tr.bpm);
      return `${lo}-${lo + 4} BPM is working on this floor`;
    }
    return "Unexplored - high danceability, safe tempo";
  }

  /* ---- the two, compared ------------------------------------------------ */
  loopCompare(n = 3, count = false) {
    const closed = this.recommend(n);
    const open = this.recommendOpen(n);
    const base = {
      open, closed, corrected: this.divergences, compared: this.comparisons,
      agree: null, rankShift: null, note: null,
    };
    if (!closed.length || !open.length) return base;
    const c0 = closed[0], o0 = open[0];
    const agree = c0.title === o0.title;
    const order = this.recommendOpen(CATALOGUE.length).map((t) => t.title);
    const rankShift = order.indexOf(c0.title);
    if (count) {
      this.comparisons += 1;
      if (!agree) this.divergences += 1;
    }
    return {
      ...base, agree, rankShift,
      corrected: this.divergences, compared: this.comparisons,
      note: this._note(c0, o0, agree, rankShift),
    };
  }

  _note(closed, open, agree, rankShift) {
    if (agree) return "Both models agree - the safe pick is also the right one here";
    if (!this.genreReward.size) return "No crowd reward yet - the loop has nothing to correct with";
    const gap = rankShift > 0 ? ` (#${rankShift + 1} on audio features alone)` : "";
    // A divergence is not automatically evidence of learning: if the winning
    // genre has no reward yet, the loop is exploring, and it should say so.
    if (!this._reward(this.genreReward, closed.genre).length) {
      return `Exploring: ${closed.genre} is untested on this floor tonight${gap}`;
    }
    return `Crowd response overrides the feature match: ${closed.genre} over ${open.genre}${gap}`;
  }

  genreScores() {
    const rows = [];
    for (const [genre, vals] of this.genreReward) {
      if (vals.length) rows.push({ genre, score: +mean(vals).toFixed(1), tracks: vals.length });
    }
    rows.sort((a, b) => b.score - a.score);
    return rows;
  }
}
