"""Closed-loop DJ intelligence.

THE IDEA
--------
A music recommender is an **open loop**. Spotify-style engines are extremely
good at one question -- *given this track, what sounds like it?* -- and they
answer it from audio features (energy, danceability, valence, tempo, key) plus
what millions of other listeners did next. What they can never observe is the
only thing that matters in a room: **whether the floor actually moved.**

Kadenz closes that loop. The same catalogue, the same audio features, but the
ranking is corrected by a reward signal nobody else has -- crowd response
measured by computer vision, live, from the room itself:

    catalogue (audio features)
             |
             v
        track plays  ->  vision measures the crowd  ->  reward
             ^                                            |
             |____________ profile updates _______________|

Both rankings are computed here, side by side, on every tick:

  * `recommend_open(n)`   -- CONTENT ONLY. Feature similarity to what is
                             playing, plus a popularity prior. This is the
                             open-loop baseline: it never sees the room.
  * `recommend(n)`        -- CLOSED LOOP. The same candidates re-ranked by the
                             crowd response this floor actually gave to each
                             genre and tempo band tonight.

The interesting number is not either pick on its own -- it is how often they
**disagree**, and who was right. That divergence is the product.

WHAT IS SIMULATED
-----------------
The catalogue below is SYNTHETIC: fictional tracks carrying realistic audio
features on the standard streaming-API scales. It exists so the recommender can
be demonstrated without a licensed catalogue. Swap `CATALOGUE` for a real
provider's response and none of the logic changes. The *crowd response driving
the loop is measured for real*; only the music metadata is stand-in, and it is
flagged as such everywhere it surfaces.
"""
import random
from collections import defaultdict

# --- Synthetic catalogue -----------------------------------------------------
# Feature names and scales mirror the standard audio-feature schema used by
# streaming catalogue APIs: energy / danceability / valence in 0-1, tempo in
# BPM, musical key, plus a popularity prior in 0-1. `pop` is what drives an
# open-loop recommender towards the safe, globally-liked pick -- and it is
# exactly what a specific room at 2am does not care about.
CATALOGUE = [
    {"title": "Neon Tide", "artist": "Vela Nox", "genre": "Melodic Techno",
     "bpm": 124, "key": "A min", "energy": 0.68, "dance": 0.79, "valence": 0.32, "pop": 0.62},
    {"title": "Concrete Bloom", "artist": "Ferrite", "genre": "Melodic Techno",
     "bpm": 126, "key": "F min", "energy": 0.74, "dance": 0.81, "valence": 0.28, "pop": 0.55},
    {"title": "Halogen", "artist": "Kestrel Park", "genre": "Melodic Techno",
     "bpm": 128, "key": "C min", "energy": 0.80, "dance": 0.83, "valence": 0.35, "pop": 0.71},
    {"title": "Mercury Drift", "artist": "Alva Sound", "genre": "Deep House",
     "bpm": 120, "key": "G min", "energy": 0.55, "dance": 0.86, "valence": 0.48, "pop": 0.78},
    {"title": "Late Bloomer", "artist": "Marisol Vane", "genre": "Deep House",
     "bpm": 122, "key": "D min", "energy": 0.60, "dance": 0.88, "valence": 0.52, "pop": 0.83},
    {"title": "Paper Lantern", "artist": "Oto Kin", "genre": "Deep House",
     "bpm": 123, "key": "A min", "energy": 0.63, "dance": 0.85, "valence": 0.45, "pop": 0.66},
    {"title": "Iron Garden", "artist": "Bruk Theory", "genre": "Peak Techno",
     "bpm": 134, "key": "E min", "energy": 0.92, "dance": 0.76, "valence": 0.22, "pop": 0.49},
    {"title": "Afterburn", "artist": "Null Sector", "genre": "Peak Techno",
     "bpm": 138, "key": "B min", "energy": 0.96, "dance": 0.72, "valence": 0.18, "pop": 0.41},
    {"title": "Strobe Church", "artist": "Hex Lumen", "genre": "Peak Techno",
     "bpm": 136, "key": "F min", "energy": 0.94, "dance": 0.74, "valence": 0.20, "pop": 0.44},
    {"title": "Sunday Chrome", "artist": "Petra Lune", "genre": "Disco House",
     "bpm": 118, "key": "C maj", "energy": 0.66, "dance": 0.91, "valence": 0.81, "pop": 0.88},
    {"title": "Velvet Hours", "artist": "Club Soleil", "genre": "Disco House",
     "bpm": 121, "key": "G maj", "energy": 0.70, "dance": 0.93, "valence": 0.85, "pop": 0.91},
    {"title": "Paloma", "artist": "Rue Atlas", "genre": "Afro House",
     "bpm": 122, "key": "D min", "energy": 0.72, "dance": 0.90, "valence": 0.64, "pop": 0.74},
    {"title": "Dust and Salt", "artist": "Kaya Mbeki", "genre": "Afro House",
     "bpm": 124, "key": "A min", "energy": 0.76, "dance": 0.92, "valence": 0.68, "pop": 0.69},
    {"title": "Low Ceiling", "artist": "Grain Index", "genre": "Breakbeat",
     "bpm": 140, "key": "E min", "energy": 0.85, "dance": 0.78, "valence": 0.40, "pop": 0.38},
    {"title": "Pirate Radio", "artist": "Slate Runner", "genre": "Breakbeat",
     "bpm": 142, "key": "G min", "energy": 0.88, "dance": 0.75, "valence": 0.44, "pop": 0.35},
    {"title": "Blue Hour", "artist": "Ilma Reyes", "genre": "Ambient House",
     "bpm": 112, "key": "F maj", "energy": 0.35, "dance": 0.60, "valence": 0.55, "pop": 0.72},
    {"title": "Glass Field", "artist": "Noor Ateles", "genre": "Ambient House",
     "bpm": 115, "key": "C maj", "energy": 0.40, "dance": 0.64, "valence": 0.58, "pop": 0.64},
]

MIXABLE_BPM_PCT = 0.06   # a DJ beatmatches roughly +/-6% without artefacts

# Feature weights for open-loop content similarity. Tempo is normalised over the
# 108-146 BPM range the catalogue spans.
_SIM_W = {"energy": 0.30, "dance": 0.30, "valence": 0.15, "tempo": 0.25}
_BPM_LO, _BPM_HI = 108.0, 146.0


def _tempo_norm(bpm):
    return (float(bpm) - _BPM_LO) / (_BPM_HI - _BPM_LO)


def feature_distance(a, b):
    """Weighted L1 distance between two tracks in audio-feature space (0-1)."""
    d = (_SIM_W["energy"] * abs(a["energy"] - b["energy"])
         + _SIM_W["dance"] * abs(a["dance"] - b["dance"])
         + _SIM_W["valence"] * abs(a.get("valence", 0.5) - b.get("valence", 0.5))
         + _SIM_W["tempo"] * abs(_tempo_norm(a["bpm"]) - _tempo_norm(b["bpm"])))
    return min(1.0, d)


class DJEngine:
    """Learns which audio features move THIS floor and picks what to play next.

    Contextual-bandit style: every track played leaves a reward (the crowd
    response it produced). Genres and tempo bands accumulate those rewards; the
    next pick maximises expected response, constrained by mixability and with a
    small exploration term so the set does not collapse into a single genre.

    The class also carries the open-loop baseline it is meant to beat, so the
    two can be compared live rather than argued about.
    """

    def __init__(self, seed=7):
        self.rng = random.Random(seed)
        self.history = []                       # finished tracks + their response
        self.genre_reward = defaultdict(list)   # genre -> [mean crowd energy]
        self.tempo_reward = defaultdict(list)   # bpm band -> [mean crowd energy]
        self.current = None
        self.played_titles = set()
        self.divergences = 0                    # track changes where picks differed
        self.comparisons = 0

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _band(bpm):
        return int(bpm // 5) * 5   # 5-BPM buckets: 120-124, 125-129, ...

    @staticmethod
    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def _baseline(self):
        """Average crowd response across the session so far."""
        scored = [h["response"] for h in self.history if h["response"] is not None]
        return self._mean(scored) if scored else None

    def _candidates(self):
        """Everything playable right now: the catalogue minus what is spinning."""
        return [tr for tr in CATALOGUE
                if not (self.current and tr["title"] == self.current["title"])]

    def _mixable(self, tr):
        """Can a DJ beatmatch this from what is currently playing?"""
        cur_bpm = self.current["bpm"] if self.current else None
        if not cur_bpm:
            return True, 0.0
        drift = abs(tr["bpm"] - cur_bpm) / cur_bpm
        return drift <= MIXABLE_BPM_PCT, drift

    # ------------------------------------------------------------------ state
    def start_track(self, track, t, open_pick=None):
        """Put a track on. `open_pick` records what the open loop wanted instead."""
        self.current = {**track, "start": t, "samples": [], "open_pick": open_pick}
        self.played_titles.add(track["title"])
        return self.current

    def sample(self, energy):
        """Feed the live crowd energy measured by the vision pipeline."""
        if self.current is not None:
            self.current["samples"].append(float(energy))

    def close_current(self, t):
        """Finish the current track, scoring the response it produced."""
        cur = self.current
        if cur is None:
            return None
        response = self._mean(cur["samples"]) if cur["samples"] else None
        base = self._baseline()
        delta = None if (response is None or base is None or base <= 0) else \
            100.0 * (response - base) / base
        entry = {k: cur[k] for k in
                 ("title", "artist", "genre", "bpm", "key", "energy", "dance")}
        entry["valence"] = cur.get("valence")
        entry.update({"response": response, "delta": delta,
                      "start": cur["start"], "end": t,
                      "open_pick": cur.get("open_pick")})
        self.history.append(entry)
        if response is not None:
            self.genre_reward[entry["genre"]].append(response)
            self.tempo_reward[self._band(entry["bpm"])].append(response)
        self.current = None
        return entry

    # ----------------------------------------------------- OPEN LOOP baseline
    def recommend_open(self, n=3):
        """Content-only ranking: what a recommender that cannot see the room picks.

        Pure audio-feature similarity to the track playing, plus a popularity
        prior -- the standard seed-track approach. No crowd signal enters here;
        that is the entire point. With nothing playing yet it falls back to
        popularity, exactly like a cold-start recommendation.
        """
        cur = self.current
        out = []
        for tr in self._candidates():
            if cur is not None:
                sim = 1.0 - feature_distance(cur, tr)
                score = 100.0 * (0.70 * sim + 0.30 * tr["pop"])
            else:
                score = 100.0 * tr["pop"]
            if tr["title"] in self.played_titles:
                score -= 25.0
            mixable, _ = self._mixable(tr)
            out.append({**tr, "score": round(score, 1), "mixable": mixable,
                        "reason": self._reason_open(tr, cur)})
        out.sort(key=lambda x: -x["score"])
        return out[:n]

    def _reason_open(self, tr, cur):
        if cur is None:
            return "Highest popularity in catalogue ({:.0%})".format(tr["pop"])
        sim = 1.0 - feature_distance(cur, tr)
        return "{:.0%} audio-feature match, popularity {:.0%}".format(sim, tr["pop"])

    # --------------------------------------------------- CLOSED LOOP (Kadenz)
    def recommend(self, n=3):
        """Rank the catalogue by expected crowd response for THIS floor."""
        global_mean = self._mean([r for rs in self.genre_reward.values() for r in rs])
        out = []
        for tr in self._candidates():
            g = self.genre_reward.get(tr["genre"], [])
            b = self.tempo_reward.get(self._band(tr["bpm"]), [])
            # Expected response: what this floor gave to that genre / tempo band.
            # Unknowns fall back to the session mean (neutral prior).
            g_score = self._mean(g) if g else global_mean
            b_score = self._mean(b) if b else global_mean
            score = 0.55 * g_score + 0.30 * b_score + 15.0 * tr["dance"]
            # Mixability. This has to be a real constraint, not a nudge: a track
            # the DJ cannot beatmatch is unusable however well the floor would
            # have received it. A fixed cliff at the threshold plus a steep ramp
            # past it means only an overwhelming reward gap can override it.
            mixable, drift = self._mixable(tr)
            if not mixable:
                score -= 30.0 + 300.0 * (drift - MIXABLE_BPM_PCT)
            if tr["title"] in self.played_titles:
                score -= 25.0                      # do not repeat the set
            score += self.rng.uniform(0, 4)        # exploration
            out.append({**tr, "score": round(score, 1), "mixable": mixable,
                        "reason": self._reason(tr, g, b)})
        out.sort(key=lambda x: -x["score"])
        return out[:n]

    def _reason(self, tr, g, b):
        """One human sentence explaining why this track is suggested."""
        if g:
            avg = self._mean(g)
            base = self._baseline()
            if base and base > 0:
                d = 100.0 * (avg - base) / base
                if d >= 5:
                    return "{} running {:+.0f}% above tonight's average".format(tr["genre"], d)
                if d <= -5:
                    return "{} under-performed {:+.0f}% - risky pick".format(tr["genre"], d)
            return "{} matches tonight's profile".format(tr["genre"])
        if b:
            lo = self._band(tr["bpm"])
            return "{}-{} BPM is working on this floor".format(lo, lo + 4)
        return "Unexplored - high danceability, safe tempo"

    # ------------------------------------------------------ the two, compared
    def loop_compare(self, n=3, count=False):
        """Both rankings and the divergence between them.

        Returns the open-loop pick, the closed-loop pick, whether they agree,
        and how far the closed-loop choice sits down the open-loop list -- i.e.
        how big a leap the crowd signal is asking the DJ to take. Set
        `count=True` once per track change to fold the result into the running
        divergence rate (so a 2 Hz refresh does not inflate the statistic).
        """
        closed = self.recommend(n)
        openl = self.recommend_open(n)
        base = {"open": openl, "closed": closed,
                "corrected": self.divergences, "compared": self.comparisons}
        if not closed or not openl:
            base.update({"agree": None, "rank_shift": None,
                         "divergence_pct": None, "note": None})
            return base
        c0, o0 = closed[0], openl[0]
        agree = c0["title"] == o0["title"]
        # Where does the closed-loop winner sit in the open-loop ranking?
        order = [t["title"] for t in self.recommend_open(len(CATALOGUE))]
        rank_shift = order.index(c0["title"]) if c0["title"] in order else None
        if count:
            self.comparisons += 1
            if not agree:
                self.divergences += 1
        pct = (100.0 * self.divergences / self.comparisons) if self.comparisons else None
        base.update({
            "agree": agree, "rank_shift": rank_shift,
            "divergence_pct": None if pct is None else round(pct),
            "corrected": self.divergences, "compared": self.comparisons,
            "note": self._divergence_note(c0, o0, agree, rank_shift),
        })
        return base

    def _divergence_note(self, closed, openl, agree, rank_shift):
        """Plain sentence describing what the crowd signal changed.

        Careful here: a divergence is not automatically evidence of learning. If
        the winning genre has no reward yet, the loop is *exploring*, not
        correcting, and saying otherwise would overclaim the very thing this
        project is trying to demonstrate.
        """
        if agree:
            return "Both models agree - the safe pick is also the right one here"
        if not self.genre_reward:
            return "No crowd reward yet - the loop has nothing to correct with"
        gap = ""
        if rank_shift is not None and rank_shift > 0:
            gap = " (#{} on audio features alone)".format(rank_shift + 1)
        if not self.genre_reward.get(closed["genre"]):
            return "Exploring: {} is untested on this floor tonight{}".format(
                closed["genre"], gap)
        return "Crowd response overrides the feature match: {} over {}{}".format(
            closed["genre"], openl["genre"], gap)

    # -------------------------------------------------------------- reporting
    def genre_scores(self):
        """Per-genre crowd response, best first - the 'what works here' table."""
        rows = []
        for genre, vals in self.genre_reward.items():
            if vals:
                rows.append({"genre": genre, "score": round(self._mean(vals), 1),
                             "tracks": len(vals)})
        rows.sort(key=lambda r: -r["score"])
        return rows

    def best_track(self):
        scored = [h for h in self.history if h["response"] is not None]
        return max(scored, key=lambda h: h["response"]) if scored else None
