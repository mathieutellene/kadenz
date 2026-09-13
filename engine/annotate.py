"""Overlay renderer.

Design rules (the video must stay readable underneath):
  * everything is drawn on a separate layer and alpha-blended, never painted
    solid over the footage;
  * skeletons are hairline strokes, not boxes -- the bounding box is reduced to
    four short corner ticks;
  * a limb is only drawn when BOTH of its joints are confident, so half-detected
    people do not produce nonsense geometry;
  * per-person text is limited to the few highest-energy dancers, because the
    numbers live in the dashboard, not on top of the crowd.

Colour encodes energy on an audio-software ramp: cyan (still) -> amber -> hot
pink (going off).
"""
import cv2
import numpy as np

# BGR ramp stops: cyan -> amber -> hot pink
C_COLD = np.array([238, 210, 34], dtype=float)    # #22d2ee cyan
C_WARM = np.array([36, 191, 251], dtype=float)    # #fbbf24 amber
C_HOT = np.array([94, 62, 244], dtype=float)      # #f43e5e hot pink
C_IDLE = (150, 140, 160)
TEXT = (245, 242, 252)

# COCO-17 skeleton, grouped so a missing limb never links unrelated joints
EDGES = (
    (5, 7), (7, 9),            # left arm
    (6, 8), (8, 10),           # right arm
    (5, 6),                    # shoulders
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),        # left leg
    (12, 14), (14, 16),        # right leg
    (0, 5), (0, 6),            # neck
)
KP_CONF = 0.45          # stricter than default: kills ghost limbs
OVERLAY_ALPHA = 0.72    # how much of the overlay survives the blend
LABEL_TOP_N = 3         # only the most energetic dancers get a tag


def energy_color(energy):
    """Map 0-100 energy to the cyan->amber->pink ramp (BGR tuple)."""
    if energy is None:
        return C_IDLE
    t = max(0.0, min(1.0, energy / 100.0))
    if t < 0.5:
        c = C_COLD + (C_WARM - C_COLD) * (t / 0.5)
    else:
        c = C_WARM + (C_HOT - C_WARM) * ((t - 0.5) / 0.5)
    return tuple(int(v) for v in c)


def _fmt_mmss(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class Annotator:
    def draw(self, frame, tracks, mag=None, heat=False):
        out = frame.copy()
        s = max(out.shape[:2]) / 960.0
        # Antialiasing is lovely and expensive. On a packed floor (30+ skeletons
        # at 18 fps) it is what starves the display loop, so drop it there.
        aa = cv2.LINE_AA if len(tracks or []) <= 12 else cv2.LINE_8

        if heat and mag is not None:
            norm = np.clip(mag / (np.percentile(mag, 97) + 1e-6), 0.0, 1.0)
            m8 = cv2.resize((norm * 255).astype("uint8"), (out.shape[1], out.shape[0]))
            heatmap = cv2.applyColorMap(m8, cv2.COLORMAP_TURBO)
            mask = m8 > 40
            out[mask] = cv2.addWeighted(out, 0.55, heatmap, 0.45, 0)[mask]

        tracks = tracks or []
        if not tracks:
            return out

        layer = out.copy()
        for ti in tracks:
            color = energy_color(ti.get("e"))
            self._corner_ticks(layer, ti, color, s, aa)
            self._skeleton(layer, ti.get("kpts"), ti.get("kconf"), color, s, aa)
        out = cv2.addWeighted(layer, OVERLAY_ALPHA, out, 1.0 - OVERLAY_ALPHA, 0)

        # Labels go on last, fully opaque, and only for the top movers.
        ranked = sorted((t for t in tracks if t.get("e") is not None),
                        key=lambda t: -t["e"])[:LABEL_TOP_N]
        for ti in ranked:
            self._tag(out, ti, energy_color(ti.get("e")), s)
        return out

    # ------------------------------------------------------------------ parts
    def _corner_ticks(self, img, ti, color, s, aa=cv2.LINE_AA):
        """Four short ticks instead of a full box: marks the person, hides nothing."""
        x1, y1, x2, y2 = ti["x1"], ti["y1"], ti["x2"], ti["y2"]
        th = max(1, round(1.4 * s))
        ln = max(4, min(int(min(x2 - x1, y2 - y1) * 0.18), round(13 * s)))
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * ln, cy), color, th, aa)
            cv2.line(img, (cx, cy), (cx, cy + dy * ln), color, th, aa)

    def _skeleton(self, img, kpts, kconf, color, s, aa=cv2.LINE_AA):
        if kpts is None or kconf is None:
            return
        th = max(1, round(1.5 * s))
        for a, b in EDGES:
            if kconf[a] > KP_CONF and kconf[b] > KP_CONF:
                cv2.line(img,
                         (int(kpts[a][0]), int(kpts[a][1])),
                         (int(kpts[b][0]), int(kpts[b][1])),
                         color, th, aa)
        r = max(1, round(1.8 * s))
        for j in range(17):
            if kconf[j] > KP_CONF:
                cv2.circle(img, (int(kpts[j][0]), int(kpts[j][1])), r, TEXT, -1, aa)

    def _tag(self, img, ti, color, s):
        """Small energy tag for a notable dancer."""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.34 * s
        th = max(1, round(s))
        text = f"{ti['id']:02d} E{ti['e']}"
        (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
        pad = max(2, round(3 * s))
        x = max(0, min(ti["x1"], w - tw - 2 * pad))
        y2 = max(tht + 2 * pad, ti["y1"] - max(2, round(3 * s)))
        y1 = y2 - (tht + 2 * pad)
        cv2.rectangle(img, (x, y1), (x + tw + 2 * pad, y2), (18, 14, 24), -1)
        cv2.line(img, (x, y2), (x + tw + 2 * pad, y2), color, max(1, round(s)), cv2.LINE_AA)
        cv2.putText(img, text, (x + pad, y2 - pad), font, fs, TEXT, th, cv2.LINE_AA)
