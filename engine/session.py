"""Analysis session: display and analysis DECOUPLED into separate threads.

Three independent clocks so the video stays smooth on any machine:
  - DISPLAY (main): decodes, scales once, draws the last known state and
    publishes frames at send_fps. It never waits for the model.
  - ANALYSIS: always grabs the most recent frame and runs YOLO + optical flow as
    fast as the CPU allows, updating shared state.
  - AUDIO: extracts BPM/level/drops in parallel; the video never waits for it.

Queue messages (dicts): hello / status / audio / metrics / dj / tracks / event /
end / error. Frames do NOT go through the queue: they live in `latest_frame`
(latest wins) and the server sends them as binary.
"""
import queue
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np

cv2.setNumThreads(2)

from .annotate import Annotator
from .audio import extract_audio_features, recognize_file_segment
from .detector import PersonTracker
from .dj import DJEngine
from .flow import FlowEnergy
from .metrics import GrooveSync, MetricsEngine, PersonalEnergy
from .prep import ensure_analyzable
from .sources import VideoSource


# How long the overlay keeps the last known skeletons through a detection gap.
TRACK_HOLD_SECONDS = 1.0


class AnalysisSession(threading.Thread):
    def __init__(self, source_spec, cfg, out_queue, name="video"):
        super().__init__(daemon=True)
        self.spec = source_spec
        self.cfg = cfg
        self.out = out_queue
        self.display_name = name
        self.stop_flag = threading.Event()
        self.heat = False
        self.latest_frame = None  # (t_ms, jpeg_bytes) -> binary WS, latest wins

        # Estado compartido display <-> analisis (swaps de referencia, atomicos por el GIL)
        self._for_analysis = None  # (t, display_frame), most recent
        self._track_state = []     # per-person dicts (box, energy, kpts...)
        self._track_state_t = -99.0   # video time the overlay was last refreshed
        self._latest_mag = None
        # (crowd, per_track, occupancy) SIEMPRE del mismo analisis — evita que
        # participacion se calcule mezclando dos fotogramas distintos (>100%)
        self._vision = (None, None, 0)
        self._proc_fps = 0.0
        self._churn_lock = threading.Lock()
        self._churn_new = 0
        self._churn_lost = 0
        self.drops = set()
        self.current_energy = 0.0  # latest global energy (drives the DJ engine)
        self.current_t = 0.0
        self._extra = {}   # derived metrics from the analyzer (speed, spread...)
        self.groove = GrooveSync()
        # Closed-loop DJ intelligence + simulated set (see engine/dj.py)
        self.dj = DJEngine()
        self.audio_db = None      # dBFS curve of the file's audio
        self.audio_rate = 10
        self.live_db = None       # dBFS from the live capture

    def emit(self, msg):
        try:
            self.out.put_nowait(msg)
        except queue.Full:
            try:
                self.out.get_nowait()
                self.out.put_nowait(msg)
            except Exception:
                pass

    def stop(self):
        self.stop_flag.set()

    # ------------------------------------------------------------------ run
    def run(self):
        vcfg = self.cfg["video"]
        mcfg = self.cfg["metrics"]
        try:
            source_spec = self.spec
            is_file = (isinstance(self.spec, str)
                       and not str(self.spec).lower().startswith(("http", "rtsp"))
                       and not str(self.spec).isdigit())
            if is_file:
                source_spec = ensure_analyzable(
                    self.spec,
                    on_status=lambda t: self.emit({"type": "status", "text": t}),
                )
            src = VideoSource(source_spec).open()
        except Exception as e:
            self.emit({"type": "error", "text": str(e)})
            return

        detector = PersonTracker(
            self.cfg["model"]["weights"], vcfg["det_conf"], vcfg["det_imgsz"], src.fps
        )
        flow = FlowEnergy(vcfg["flow_width"])
        personal = PersonalEnergy()
        metrics = MetricsEngine(mcfg["ema_alpha"], mcfg["peak_z"], mcfg["peak_min_gap_s"])
        annot = Annotator()

        self.emit({
            "type": "hello",
            "video": {
                "name": self.display_name,
                "fps": round(src.fps, 2),
                "width": src.width,
                "height": src.height,
                "duration": src.duration,
            },
            "audio": None,  # llegara en un mensaje "audio" cuando este listo
        })

        if isinstance(self.spec, str) and not src.is_stream:
            threading.Thread(target=self._audio_job, daemon=True).start()
        elif src.is_stream:
            try:
                from .audio_live import LiveAudio
                LiveAudio(self).start()
            except Exception:
                self.emit({"type": "status", "text": "Audio en directo no disponible"})
        analyzer = threading.Thread(
            target=self._analyzer_loop, args=(detector, flow, personal), daemon=True
        )
        analyzer.start()

        try:
            self._display_loop(src, metrics, annot, vcfg, mcfg)
        except Exception:
            self.emit({"type": "error", "text": traceback.format_exc(limit=8)})
        finally:
            self.stop_flag.set()
            src.release()
            self.emit({"type": "end", "summary": metrics.summary()})

    # ------------------------------------------------------------- audio job
    def _audio_job(self):
        self.emit({"type": "status", "text": "Analizando el audio en segundo plano..."})
        audio = extract_audio_features(self.spec)
        if self.stop_flag.is_set():
            return
        if audio:
            self.drops.update(audio.get("drops", []))
            self.audio_db = audio.get("db")
            self.audio_rate = audio.get("rate", 10)
            self.emit({"type": "audio", **audio})
            rate = audio.get("rate", 10)
            for i, v in enumerate(audio.get("onset") or []):  # semilla del Groove Sync
                self.groove.add_audio(i / rate, v)
            info = recognize_file_segment(self.spec)
            if not self.stop_flag.is_set():
                if info:
                    self.emit({"type": "event", "kind": "track", "t": round(self.current_t, 1),
                               "text": f"Suena: {info['artist']} – {info['title']} ({info['genre']})"})
                    self.emit({"type": "now_playing", **info, "avg_e": None, "best": None})
                else:
                    self.emit({"type": "now_playing", "title": None})
        else:
            self.emit({"type": "status", "text": "Este video no tiene audio analizable"})

    # ------------------------------------------------------- level + DJ loop
    def _db_at(self, t):
        """Sound pressure level at time t, in dB SPL (the scale a DJ works in).

        A laptop mic is not a calibrated SPL meter, so measured audio is mapped
        from dBFS with a fixed offset and reported as an ESTIMATE; club systems
        typically run 95-110 dB SPL on the floor.
        """
        offset = float((self.cfg.get("audio", {}) or {}).get("spl_offset", 108.0))
        dbfs = None
        if self.live_db is not None:
            dbfs = float(self.live_db)
        elif self.audio_db:
            i = int(t * self.audio_rate)
            if 0 <= i < len(self.audio_db):
                dbfs = float(self.audio_db[i])
        if dbfs is not None:
            if dbfs < -60.0:
                dbfs = None          # digital silence: fall through, do not clamp
            else:
                return round(max(40.0, min(120.0, dbfs + offset)), 1)
        cur = self.dj.current
        if cur is not None:
            # Simulated PA level: harder tracks push the room louder, and the
            # measured crowd energy nudges it a little.
            base = 93.0 + 14.0 * float(cur.get("energy", 0.6))
            return round(base + 0.04 * (self.current_energy - 50.0), 1)
        return None

    def _bpm_now(self):
        """Tempo of the simulated track, when there is no real audio to measure."""
        cur = self.dj.current
        return None if cur is None else cur.get("bpm")

    def _emit_dj(self, t, changed=False):
        cur = self.dj.current
        now = None
        if cur is not None:
            resp = self.dj._mean(cur["samples"]) if cur["samples"] else None
            now = {k: cur[k] for k in
                   ("title", "artist", "genre", "bpm", "key", "energy",
                    "dance", "valence")}
            now["elapsed"] = round(t - cur["start"], 1)
            now["response"] = None if resp is None else round(resp, 1)
        last = self.dj.history[-1] if self.dj.history else None
        # Both rankings go out on every push -- the content-only open-loop
        # baseline and the closed-loop pick -- so the UI can show them side by
        # side instead of asking anyone to take our word for the difference.
        # Ranked live, so the panel always proposes the NEXT track (whatever is
        # on the deck is excluded from the candidates). The running divergence
        # tally is a separate, stable statistic, incremented only in _dj_tick at
        # the moment a decision is actually taken.
        cmp_ = self.dj.loop_compare(3)
        self.emit({
            "type": "dj",
            "simulated": True,
            "changed": changed,
            "now": now,
            "next": cmp_["closed"],
            "open": cmp_["open"],
            "loop": {"agree": cmp_["agree"], "rank_shift": cmp_["rank_shift"],
                     "divergence_pct": cmp_["divergence_pct"],
                     "corrected": cmp_["corrected"], "compared": cmp_["compared"],
                     "note": cmp_["note"]},
            "genres": self.dj.genre_scores(),
            "last": None if last is None else {
                "title": last["title"], "genre": last["genre"],
                "delta": None if last["delta"] is None else round(last["delta"], 1),
                "open_pick": last.get("open_pick")},
        })

    def _dj_tick(self, t, energy):
        """Run the simulated set: feed crowd response, rotate tracks, recommend.

        The music side is SIMULATED (no licensed catalogue); the crowd response
        driving it is measured for real by the vision pipeline.
        """
        cfg = self.cfg.get("dj", {}) or {}
        if not cfg.get("simulate", True):
            return
        span = float(cfg.get("track_seconds", 20))
        self.dj.sample(energy)
        cur = self.dj.current
        if cur is None or (t - cur["start"]) >= span:
            if cur is not None:
                self.dj.close_current(t)
            # One comparison per track change (not per refresh) so the running
            # divergence rate counts decisions, not UI ticks.
            cmp_ = self.dj.loop_compare(3, count=True)
            if cmp_["closed"]:
                op = cmp_["open"][0] if cmp_["open"] else None
                self.dj.start_track(
                    cmp_["closed"][0], t,
                    open_pick=None if op is None else
                    {"title": op["title"], "artist": op["artist"],
                     "genre": op["genre"], "bpm": op["bpm"],
                     "reason": op["reason"]})
            self._emit_dj(t, changed=True)
            self._dj_last_push = t
        elif t - getattr(self, "_dj_last_push", -99) >= 2.0:
            self._emit_dj(t, changed=False)
            self._dj_last_push = t

    # ---------------------------------------------------------- display loop
    def _display_loop(self, src, metrics, annot, vcfg, mcfg):
        frame_idx = 0
        t0_wall = time.perf_counter()
        last_sent = -1e9
        last_metric_t = 0.0
        send_dt = 1.0 / vcfg["send_fps"]
        metric_dt = 1.0 / mcfg["rate_hz"]
        dw = vcfg["display_width"]
        announced_drops = set()

        while not self.stop_flag.is_set():
            ok, frame = src.read()
            if not ok:
                break
            # Archivos: reloj del video (idx/fps). Streams/webcam: reloj de pared,
            # porque el fps que reporta una webcam no es fiable.
            if src.is_stream:
                t_video = time.perf_counter() - t0_wall
            else:
                t_video = frame_idx / src.fps
            self.current_t = t_video

            if not src.is_stream:
                ahead = t_video - (time.perf_counter() - t0_wall)
                if ahead > 0.005:
                    time.sleep(min(ahead, 0.25))
                elif ahead < -0.35:
                    skip = int(-ahead * src.fps) - 2
                    for _ in range(max(0, skip)):
                        if not src.grab():
                            break
                        frame_idx += 1
                    t_video = frame_idx / src.fps

            # Reescala UNA vez por el lado largo; todo aguas abajo usa este tamano.
            long_side = max(frame.shape[:2])
            if long_side > dw:
                s = dw / long_side
                frame = cv2.resize(
                    frame,
                    (max(2, int(frame.shape[1] * s)), max(2, int(frame.shape[0] * s))),
                    interpolation=cv2.INTER_AREA,
                )
            self._for_analysis = (t_video, frame)

            if t_video - last_metric_t >= metric_dt:
                with self._churn_lock:
                    n_new, n_lost = self._churn_new, self._churn_lost
                    self._churn_new = self._churn_lost = 0
                crowd_v, per_track_v, occupancy_v = self._vision
                self._vision = (None, None, occupancy_v)  # consumidos; metricas mantienen valor
                snap, event = metrics.update(
                    t_video, crowd_v, per_track_v, occupancy_v, n_new, n_lost
                )
                last_metric_t = t_video
                snap["proc_fps"] = round(self._proc_fps, 1)
                snap["db"] = self._db_at(t_video)
                if self.audio_db is None and self.live_db is None:
                    snap["bpm"] = self._bpm_now()   # from the simulated set
                    snap["sim_audio"] = True
                snap["groove"] = self.groove.compute(t_video)
                snap.update(self._extra)
                self.current_energy = snap["energy"]
                self.emit({"type": "metrics", **snap})
                self._dj_tick(t_video, snap["energy"])
                if event:
                    self.emit({"type": "event", **event})
                for d in sorted(self.drops):
                    if d <= t_video and d not in announced_drops:
                        announced_drops.add(d)
                        self.emit({"type": "event", "kind": "drop", "t": d,
                                   "text": "Drop en la musica"})

            if t_video - last_sent >= send_dt:
                out = annot.draw(frame, self._track_state, self._latest_mag, self.heat)
                ok_jpg, buf = cv2.imencode(
                    ".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, vcfg["jpeg_quality"]]
                )
                if ok_jpg:
                    self.latest_frame = (int(t_video * 1000), buf.tobytes())
                last_sent = t_video

            frame_idx += 1

    # --------------------------------------------------------- analyzer loop
    def _analyzer_loop(self, detector, flow, personal):
        try:  # calenton de YOLO con un frame dummy: la primera inferencia real ya es rapida
            detector.model.predict(
                np.zeros((320, 320, 3), dtype=np.uint8), imgsz=320, verbose=False
            )
        except Exception:
            pass
        last_t = None
        loop_times = deque(maxlen=20)
        prev_centroids = {}   # tid -> (cx, cy, t) para velocidad de desplazamiento
        speed_ema = None
        while not self.stop_flag.is_set():
            item = self._for_analysis
            if item is None or item[0] == last_t:
                time.sleep(0.01)
                continue
            t, frame = item
            last_t = t
            tic = time.perf_counter()
            try:
                det, n_new, n_lost = detector.infer(frame, t)
                with self._churn_lock:
                    self._churn_new += n_new
                    self._churn_lost += n_lost

                mag = flow.process(frame, t)
                crowd = None
                per_track = None
                if mag is not None:
                    boxes_small = det.xyxy * flow.scale if len(det) else None
                    crowd, per_track = flow.summarize(mag, boxes_small)
                    self._latest_mag = mag

                track_info = []
                if len(det) and det.tracker_id is not None:
                    aligned = (
                        per_track
                        if per_track is not None and len(per_track) == len(det)
                        else None
                    )
                    kpts_all = det.data.get("kpts")
                    kconf_all = det.data.get("kconf")
                    active_ids = set()
                    for i, tid_raw in enumerate(det.tracker_id):
                        tid = int(tid_raw)
                        active_ids.add(tid)
                        raw = aligned[i] if aligned is not None else None
                        energy = personal.update(tid, raw)
                        x1, y1, x2, y2 = (int(v) for v in det.xyxy[i])
                        track_info.append({
                            "id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "e": energy,
                            "ton": detector.time_on_floor(tid, t),
                            "kpts": kpts_all[i] if kpts_all is not None else None,
                            "kconf": kconf_all[i] if kconf_all is not None else None,
                        })
                    personal.prune(active_ids)
                # Hold the last overlay through a brief detection gap. The
                # display thread draws whatever is in _track_state at 18 fps, so
                # wiping it on a single empty pass makes every skeleton strobe
                # off and back on -- and on dark, backlit or crowded footage
                # empty passes are common. Stale geometry for a fraction of a
                # second is far less wrong than a flashing overlay; past the
                # hold, the floor really is empty and it clears.
                if track_info:
                    self._track_state = track_info
                    self._track_state_t = t
                elif t - self._track_state_t > TRACK_HOLD_SECONDS:
                    self._track_state = []
                self._vision = (crowd, per_track, int(len(det)))
                if crowd is not None:
                    self.groove.add_motion(t, crowd)

                # --- metricas derivadas ---
                frame_h = float(frame.shape[0])
                frame_diag = float(np.hypot(frame.shape[0], frame.shape[1]))
                centroids = {}
                speeds = []
                for ti in track_info:
                    cx = (ti["x1"] + ti["x2"]) / 2.0
                    cy = (ti["y1"] + ti["y2"]) / 2.0
                    centroids[ti["id"]] = (cx, cy)
                    prev = prev_centroids.get(ti["id"])
                    if prev is not None and t > prev[2]:
                        dist = float(np.hypot(cx - prev[0], cy - prev[1]))
                        speeds.append(dist / (t - prev[2]) / frame_h)
                prev_centroids = {tid: (c[0], c[1], t) for tid, c in centroids.items()}
                extra = {}
                if speeds:
                    raw_speed = float(np.mean(speeds))
                    speed_ema = raw_speed if speed_ema is None else 0.3 * raw_speed + 0.7 * speed_ema
                    extra["speed_pct"] = round(min(speed_ema * 100.0, 99.0), 1)
                pts = list(centroids.values())
                if len(pts) >= 3:  # compactacion: distancia media al vecino mas cercano
                    arr = np.array(pts)
                    dists = []
                    for i in range(len(arr)):
                        d = np.hypot(arr[:, 0] - arr[i, 0], arr[:, 1] - arr[i, 1])
                        d[i] = np.inf
                        dists.append(float(d.min()))
                    nn_norm = float(np.mean(dists)) / frame_diag
                    extra["proximity"] = round(float(np.clip(100.0 * (1.0 - nn_norm / 0.35), 0, 100)), 1)
                scored = [ti for ti in track_info if ti.get("e") is not None]
                if scored:
                    star = max(scored, key=lambda x: x["e"])
                    extra["star"] = {"id": star["id"], "e": star["e"]}
                tons = [ti["ton"] for ti in track_info if ti.get("ton") is not None]
                if tons:
                    extra["avg_ton"] = round(float(np.mean(tons)), 1)
                self._extra = extra

                # Vision del analisis para el mini-panel del dashboard
                h, w = frame.shape[:2]
                items = []
                for ti in track_info:
                    item = {"id": ti["id"],
                            "b": [ti["x1"], ti["y1"], ti["x2"], ti["y2"]],
                            "e": ti["e"]}
                    if ti["kpts"] is not None:
                        item["k"] = np.round(ti["kpts"], 1).tolist()
                        item["c"] = np.round(ti["kconf"], 2).tolist()
                    items.append(item)
                self.emit({"type": "tracks", "t": round(t, 2), "w": w, "h": h,
                           "items": items})

                loop_times.append(time.perf_counter() - tic)
                self._proc_fps = 1.0 / max(float(np.mean(loop_times)), 1e-3)
            except Exception:
                self.emit({"type": "error", "text": traceback.format_exc(limit=8)})
                break
