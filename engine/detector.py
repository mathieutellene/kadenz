"""People detection + pose + persistent identities.

One network (YOLOv8-pose) yields boxes AND 17 COCO keypoints per person; the
keypoints ride inside `detections.data` so ByteTrack reorders them together with
the boxes when it assigns identities.

Floor flux only counts CONFIRMED tracks: a track is born after it survives
`min_age_s` and dies after `grace_s` unseen. Without that, every tracker blink
under a strobe would read as somebody entering or leaving.
"""
import os

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO


class PersonTracker:
    def __init__(self, weights, conf=0.3, imgsz=640, frame_rate=30.0,
                 min_age_s=1.5, grace_s=2.0):
        # En CPU, torch acapara todos los nucleos y asfixia al navegador del dashboard:
        # dejamos la mitad libre.
        torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.min_age_s = min_age_s
        self.grace_s = grace_s
        try:
            self.tracker = sv.ByteTrack(frame_rate=int(round(frame_rate)))
        except TypeError:
            self.tracker = sv.ByteTrack()
        self._first_seen = {}
        self._last_seen = {}
        self._confirmed = set()

    def infer(self, frame, t):
        """Devuelve (detections, n_entradas_confirmadas, n_salidas_confirmadas).

        Si el modelo es de pose, detections.data trae "kpts" (N,17,2) y "kconf" (N,17).
        """
        result = self.model.predict(
            frame, classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        det = sv.Detections.from_ultralytics(result)
        kp = getattr(result, "keypoints", None)
        if kp is not None and kp.xy is not None and len(kp) == len(det):
            det.data["kpts"] = kp.xy.cpu().numpy().astype(np.float32)
            if kp.conf is not None:
                det.data["kconf"] = kp.conf.cpu().numpy().astype(np.float32)
            else:
                det.data["kconf"] = np.ones(det.data["kpts"].shape[:2], dtype=np.float32)
        det = self.tracker.update_with_detections(det)
        active = {int(i) for i in det.tracker_id} if det.tracker_id is not None else set()

        n_new = 0
        for tid in active:
            self._last_seen[tid] = t
            if tid not in self._first_seen:
                self._first_seen[tid] = t
            if tid not in self._confirmed and t - self._first_seen[tid] >= self.min_age_s:
                self._confirmed.add(tid)
                n_new += 1

        n_lost = 0
        for tid in list(self._confirmed):
            if t - self._last_seen.get(tid, t) > self.grace_s:
                self._confirmed.discard(tid)
                self._first_seen.pop(tid, None)
                self._last_seen.pop(tid, None)
                n_lost += 1

        for tid in list(self._last_seen):  # limpieza de blips nunca confirmados
            if tid not in self._confirmed and t - self._last_seen[tid] > self.grace_s:
                self._first_seen.pop(tid, None)
                self._last_seen.pop(tid, None)

        return det, n_new, n_lost

    def time_on_floor(self, tid, t):
        """Segundos desde que se vio por primera vez a este track (None si desconocido)."""
        t0 = self._first_seen.get(tid)
        return None if t0 is None else max(0.0, t - t0)
