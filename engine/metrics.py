"""Metrics engine: fuses perception into ~2 Hz snapshots.

Crowd energy is a percentile rank against the session's own history
("hotter than X% of what has been seen tonight"), smoothed with an EMA, so the
system auto-calibrates to any camera, distance or lens without configuration.

Also here: GrooveSync, the signature metric -- does the crowd move WITH the
music, or just move?
"""
from collections import deque

import numpy as np


def _pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / denom) if denom > 1e-9 else 0.0


class GrooveSync:
    """The signature metric: does the crowd move WITH the music, or just move?

    Pearson correlation on FIRST DIFFERENCES between motion energy and the audio
    onset envelope over a rolling window, searching the best lag within
    +/-max_lag_s to absorb camera/audio latency. Reported 0-100.
    """

    def __init__(self, window_s=25.0, max_lag_s=1.5, rate=10.0):
        self.window_s = window_s
        self.max_lag_s = max_lag_s
        self.rate = rate
        self.motion = deque(maxlen=4000)     # (t, v) at the analyzer's pace
        self.audio = deque(maxlen=36000)     # (t, v) at ~10 Hz (holds 1 h of file)

    def add_motion(self, t, v):
        self.motion.append((float(t), float(v)))

    def add_audio(self, t, v):
        self.audio.append((float(t), float(v)))

    def compute(self, t_now):
        if len(self.motion) < 25 or len(self.audio) < 60:
            return None
        t0 = t_now - self.window_s
        m = [(t, v) for t, v in self.motion if t >= t0]
        a = [(t, v) for t, v in self.audio if t0 - self.max_lag_s <= t <= t_now + self.max_lag_s]
        if len(m) < 20 or len(a) < 40:
            return None
        mt = np.array([p[0] for p in m])
        mv = np.array([p[1] for p in m])
        at = np.array([p[0] for p in a])
        av = np.array([p[1] for p in a])
        if mt[-1] - mt[0] < self.window_s * 0.5:
            return None
        grid = np.arange(max(t0, mt[0]), t_now, 1.0 / self.rate)
        if len(grid) < 40:
            return None
        mi = np.diff(np.interp(grid, mt, mv))
        best = 0.0
        for lag in np.arange(-self.max_lag_s, self.max_lag_s + 1e-9, 0.25):
            ai = np.diff(np.interp(grid + lag, at, av))
            r = _pearson(mi, ai)
            if r > best:
                best = r
        return round(100.0 * max(0.0, best), 1)


class PersonalEnergy:
    """Per-person energy, 0-100.

    EMA-smooths the flow energy inside each person's box and converts it to a
    percentile rank against ALL individual samples of the session: "this person
    is moving more than X% of what has been seen tonight".
    """

    def __init__(self, alpha=0.35, hist_len=2000, warmup=30):
        self.alpha = alpha
        self.warmup = warmup
        self.ema = {}
        self.hist = deque(maxlen=hist_len)

    def update(self, tid, raw):
        if raw is not None:
            self.hist.append(raw)
            self.ema[tid] = self.alpha * raw + (1.0 - self.alpha) * self.ema.get(tid, raw)
        val = self.ema.get(tid)
        if val is None or len(self.hist) < self.warmup:
            return None
        arr = np.asarray(self.hist, dtype=float)
        return int(round(100.0 * float((arr <= val).mean())))

    def prune(self, active_ids):
        for tid in list(self.ema):
            if tid not in active_ids:
                self.ema.pop(tid, None)


class MetricsEngine:
    def __init__(self, ema_alpha=0.35, peak_z=2.2, peak_gap_s=10.0):
        self.ema_alpha = ema_alpha
        self.peak_z = peak_z
        self.peak_gap_s = peak_gap_s
        self.raw_hist = deque(maxlen=600)  # ~5 min at 2 Hz
        self.energy_ema = 0.0
        self.last_peak_t = -1e9
        self.churn_events = deque()  # (t, entries, exits)
        self.timeline = []
        self._last_participation = 0.0

    def update(self, t, crowd_energy, per_track, occupancy, n_new, n_lost):
        if crowd_energy is not None:
            self.raw_hist.append(crowd_energy)
        hist = np.asarray(self.raw_hist, dtype=float) if self.raw_hist else np.zeros(1)
        raw = crowd_energy if crowd_energy is not None else float(hist[-1])

        if len(hist) >= 10:
            index = 100.0 * float((hist <= raw).mean())
        else:
            index = float(min(100.0, raw * 400.0))  # warm-up: rough fixed scale
        self.energy_ema = self.ema_alpha * index + (1.0 - self.ema_alpha) * self.energy_ema

        if per_track is None:
            participation = self._last_participation  # flow gated (anti-strobe): hold
        else:
            moving = 0
            if per_track:
                thr = max(float(np.percentile(per_track, 90)) * 0.25, 0.006)
                moving = sum(1 for e in per_track if e > thr)
            participation = min(100.0, 100.0 * moving / occupancy) if occupancy else 0.0
            self._last_participation = participation

        self.churn_events.append((t, n_new, n_lost))
        while self.churn_events and t - self.churn_events[0][0] > 60.0:
            self.churn_events.popleft()
        churn_in = int(sum(e[1] for e in self.churn_events))
        churn_out = int(sum(e[2] for e in self.churn_events))

        event = None
        if len(hist) >= 40:
            mu, sd = float(hist.mean()), float(hist.std()) + 1e-6
            z = (raw - mu) / sd
            if z > self.peak_z and (t - self.last_peak_t) > self.peak_gap_s:
                self.last_peak_t = t
                event = {"kind": "peak", "t": round(t, 1),
                         "text": f"Energy peak on the floor (z={z:.1f})"}

        snap = {
            "t": round(t, 2),
            "energy": round(self.energy_ema, 1),
            "energy_raw": round(index, 1),
            "occupancy": int(occupancy),
            "participation": round(participation, 1),
            "in_per_min": churn_in,
            "out_per_min": churn_out,
        }
        self.timeline.append(
            {"t": snap["t"], "energy": snap["energy"], "occupancy": snap["occupancy"]}
        )
        return snap, event

    def summary(self):
        if not self.timeline:
            return {}
        energies = [s["energy"] for s in self.timeline]
        top = sorted(self.timeline, key=lambda s: s["energy"], reverse=True)[:5]
        return {
            "duration": self.timeline[-1]["t"],
            "avg_energy": round(float(np.mean(energies)), 1),
            "max_occupancy": max(s["occupancy"] for s in self.timeline),
            "top_moments": [{"t": s["t"], "energy": s["energy"]} for s in top],
        }
