"""Motion energy from dense optical flow (Farneback) on a downscaled frame.

This layer knows nothing about people: it measures how far each pixel moved
between two frames, which is exactly why it survives darkness and occlusion
where the "smart" layers fail. Energy is weighted inside person boxes when
they exist and normalised by dt so dropped frames do not distort it.

Anti-strobe: frame pairs with a large global luminance jump are discarded.
"""
import cv2
import numpy as np


class FlowEnergy:
    def __init__(self, width=320, lum_gate=14.0):
        self.width = width
        self.lum_gate = lum_gate  # anti-estrobos: salto de luminancia global que invalida el par
        self.prev = None
        self.prev_t = None
        self.scale = 1.0

    def process(self, frame, t):
        """Magnitude map on the downscaled frame, or None if the pair was gated out."""
        h, w = frame.shape[:2]
        self.scale = self.width / float(max(h, w))  # lado largo -> mismo coste en vertical
        small = cv2.resize(
            frame, (max(2, int(w * self.scale)), max(2, int(h * self.scale)))
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mag = None
        if self.prev is not None and self.prev.shape == gray.shape:
            dt = max(t - self.prev_t, 1e-3)
            lum_jump = abs(float(gray.mean()) - float(self.prev.mean()))
            if lum_jump < self.lum_gate:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.linalg.norm(flow, axis=2)
                mag = mag / (dt * gray.shape[0])  # unidades: alturas-de-frame por segundo
        self.prev = gray
        self.prev_t = t
        return mag

    def summarize(self, mag, boxes_small):
        """(crowd_energy, per_track_energy) -- mean motion inside the person boxes."""
        H, W = mag.shape
        per_track = []
        if boxes_small is not None and len(boxes_small):
            mask = np.zeros_like(mag, dtype=bool)
            for x1, y1, x2, y2 in boxes_small:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W, int(x2)), min(H, int(y2))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True
                    per_track.append(float(mag[y1:y2, x1:x2].mean()))
                else:
                    per_track.append(0.0)
            crowd = float(mag[mask].mean()) if mask.any() else float(mag.mean())
        else:
            crowd = float(mag.mean())
        return crowd, per_track
