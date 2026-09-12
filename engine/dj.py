"""Closed-loop DJ intelligence.

Music recommenders are open loop: they know what was played, never whether the
floor emptied. This module closes the loop with the one signal nobody has --
how the crowd actually reacted, measured by computer vision in real time.

    track plays -> vision measures crowd response -> profile updates -> next track

The catalogue here is SYNTHETIC (fictional tracks with realistic audio
features). It exists so the recommendation engine can be demonstrated without a
licensed catalogue; swap `CATALOGUE` for a real provider and the logic is
unchanged. Everything produced by this module is flagged as simulated.
"""
import random
from collections import defaultdict

# --- Synthetic catalogue: fictional titles, realistic audio features ---------
# energy/dance in 0-1 (Spotify-style feature scales), bpm, musical key.
CATALOGUE = [
    {"title": "Neon Tide", "artist": "Vela Nox", "genre": "Melodic Techno",
     "bpm": 124, "key": "A min", "energy": 0.68, "dance": 0.79},
    {"title": "Concrete Bloom", "artist": "Ferrite", "genre": "Melodic Techno",
     "bpm": 126, "key": "F min", "energy": 0.74, "dance": 0.81},
    {"title": "Halogen", "artist": "Kestrel Park", "genre": "Melodic Techno",
     "bpm": 128, "key": "C min", "energy": 0.80, "dance": 0.83},
    {"title": "Mercury Drift", "artist": "Alva Sound", "genre": "Deep House",
     "bpm": 120, "key": "G min", "energy": 0.55, "dance": 0.86},
    {"title": "Late Bloomer", "artist": "Marisol Vane", "genre": "Deep House",
     "bpm": 122, "key": "D min", "energy": 0.60, "dance": 0.88},
    {"title": "Paper Lantern", "artist": "Oto Kin", "genre": "Deep House",
     "bpm": 123, "key": "A min", "energy": 0.63, "dance": 0.85},
    {"title": "Iron Garden", "artist": "Bruk Theory", "genre": "Peak Techno",
     "bpm": 134, "key": "E min", "energy": 0.92, "dance": 0.76},
    {"title": "Afterburn", "artist": "Null Sector", "genre": "Peak Techno",
     "bpm": 138, "key": "B min", "energy": 0.96, "dance": 0.72},
    {"title": "Strobe Church", "artist": "Hex Lumen", "genre": "Peak Techno",
     "bpm": 136, "key": "F min", "energy": 0.94, "dance": 0.74},
    {"title": "Sunday Chrome", "artist": "Petra Lune", "genre": "Disco House",
     "bpm": 118, "key": "C maj", "energy": 0.66, "dance": 0.91},
    {"title": "Velvet Hours", "artist": "Club Soleil", "genre": "Disco House",
     "bpm": 121, "key": "G maj", "energy": 0.70, "dance": 0.93},
    {"title": "Paloma", "artist": "Rue Atlas", "genre": "Afro House",
     "bpm": 122, "key": "D min", "energy": 0.72, "dance": 0.90},
    {"title": "Dust and Salt", "artist": "Kaya Mbeki", "genre": "Afro House",
     "bpm": 124, "key": "A min", "energy": 0.76, "dance": 0.92},
    {"title": "Low Ceiling", "artist": "Grain Index", "genre": "Breakbeat",
     "bpm": 140, "key": "E min", "energy": 0.85, "dance": 0.78},
    {"title": "Pirate Radio", "artist": "Slate Runner", "genre": "Breakbeat",
     "bpm": 142, "key": "G min", "energy": 0.88, "dance": 0.75},
    {"title": "Blue Hour", "artist": "Ilma Reyes", "genre": "Ambient House",
     "bpm": 112, "key": "F maj", "energy": 0.35, "dance": 0.60},
    {"title": "Glass Field", "artist": "Noor Ateles", "genre": "Ambient House",
     "bpm": 115, "key": "C maj", "energy": 0.40, "dance": 0.64},
]

MIXABLE_BPM_PCT = 0.06   # a DJ beatmatches roughly +/-6% without artefacts


class DJEngine:
    """Learns which audio features move THIS floor and picks what to play next.

    Contextual-bandit style: every track played leaves a reward (the crowd
    response it produced). Genres and tempo bands accumulate those rewards; the
    next pick maximises expected response, constrained by mixability and with a
    small exploration term so the set does not collapse into a single genre.
    """

    def __init__(self, seed=7):
        self.rng = random.Random(seed)
        self.history = []                       # finished tracks + their response
        self.genre_reward = defaultdict(list)   # genre -> [mean crowd energy]
        self.tempo_reward = defaultdict(list)   # bpm band -> [mean crowd energy]
        self.current = None
        self.played_titles = set()

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

    # ------------------------------------------------------------------ state
    def start_track(self, track, t):
        self.current = {**track, "start": t, "samples": []}
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
        entry.update({"response": response, "delta": delta,
                      "start": cur["start"], "end": t})
        self.history.append(entry)
        if response is not None:
            self.genre_reward[entry["genre"]].append(response)
            self.tempo_reward[self._band(entry["bpm"])].append(response)
        self.current = None
        return entry

    # --------------------------------------------------------- recommendation
    def recommend(self, n=3):
        """Rank the catalogue by expected crowd response for THIS floor."""
        cur_bpm = self.current["bpm"] if self.current else None
        global_mean = self._mean([r for rs in self.genre_reward.values() for r in rs])
        out = []
        for tr in CATALOGUE:
            if self.current and tr["title"] == self.current["title"]:
                continue
            g = self.genre_reward.get(tr["genre"], [])
            b = self.tempo_reward.get(self._band(tr["bpm"]), [])
            # Expected response: what this floor gave to that genre / tempo band.
            # Unknowns fall back to the session mean (neutral prior).
            g_score = self._mean(g) if g else global_mean
            b_score = self._mean(b) if b else global_mean
            score = 0.55 * g_score + 0.30 * b_score + 15.0 * tr["dance"]
            # Mixability: penalise anything that cannot be beatmatched cleanly.
            mixable = True
            if cur_bpm:
                drift = abs(tr["bpm"] - cur_bpm) / cur_bpm
                mixable = drift <= MIXABLE_BPM_PCT
                score -= 120.0 * max(0.0, drift - MIXABLE_BPM_PCT)
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
                    return f"{tr['genre']} running {d:+.0f}% above tonight's average"
                if d <= -5:
                    return f"{tr['genre']} under-performed {d:+.0f}% - risky pick"
            return f"{tr['genre']} matches tonight's profile"
        if b:
            lo = self._band(tr["bpm"])
            return f"{lo}-{lo + 4} BPM is working on this floor"
        return "Unexplored - high danceability, safe tempo"

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
