"""Live audio with DUAL capture (Windows).

Listens to two sources at once and picks whichever has signal:
  - SYSTEM (WASAPI loopback): whatever the computer is playing.
  - MICROPHONE: music coming from speakers in the room.
System wins when both are live (cleaner signal); source changes are announced.
Captures run in CALLBACK mode: loopback devices block read() during silence, so
no blocking reads.

Every ~2 s  -> BPM and level of the active source.
Every ~20 s -> track identification (Shazam).
"""
import asyncio
import os
import tempfile
import threading
import time
import traceback
from collections import deque

import numpy as np

try:  # shazamio decodifica con pydub: apuntarlo al ffmpeg embebido
    import pydub
    import imageio_ffmpeg
    pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

SILENCE_SYSTEM = 3e-4   # el loopback en silencio es silencio digital puro
SILENCE_MIC = 6e-4      # musica de sala puede llegar floja al micro del portatil


class _Capture:
    """Una fuente de audio (dispositivo) con su buffer circular propio."""

    def __init__(self, p, pyaudio_mod, device, max_buf_s=30.0):
        self.name = device["name"]
        self.sr = int(device["defaultSampleRate"])
        self.channels = max(1, int(device["maxInputChannels"]))
        self.lock = threading.Lock()
        self.buf = deque()
        self.buf_len = 0.0
        self.max_buf_s = max_buf_s

        def cb(in_data, frame_count, time_info, status):
            x = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            if self.channels > 1:
                x = x.reshape(-1, self.channels).mean(axis=1)
            with self.lock:
                self.buf.append(x)
                self.buf_len += len(x) / self.sr
                while self.buf_len > self.max_buf_s and self.buf:
                    old = self.buf.popleft()
                    self.buf_len -= len(old) / self.sr
            return (None, pyaudio_mod.paContinue)

        self.stream = p.open(format=pyaudio_mod.paInt16, channels=self.channels,
                             rate=self.sr, input=True,
                             input_device_index=device["index"],
                             frames_per_buffer=2048, stream_callback=cb)
        self.stream.start_stream()

    def window(self, seconds):
        with self.lock:
            if self.buf_len < 1.0:
                return None
            x = np.concatenate(list(self.buf))
        n = int(seconds * self.sr)
        return x[-n:] if len(x) > n else x

    def rms(self, seconds=2.0):
        y = self.window(seconds)
        if y is None or not len(y):
            return 0.0
        return float(np.sqrt(np.mean(y ** 2)))

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass


class LiveAudio(threading.Thread):
    def __init__(self, session, analyze_every=2.0, recognize_every=20.0):
        super().__init__(daemon=True)
        self.s = session
        self.analyze_every = analyze_every
        self.recognize_every = recognize_every
        self.cap_system = None
        self.cap_mic = None
        self._source_name = None
        self.level_hist = deque(maxlen=300)
        # Copilot
        self.tracks = []
        self.current = None
        self.unknown_energies = []
        self._reco_fails = 0
        self._reco_warned = False
        self._mid_advised = set()
        self._env_ref = deque(maxlen=200)
        self._np_counter = 0
        self._nomatch = 0

    # ------------------------------------------------------------------ hilo
    def run(self):
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            self.s.emit({"type": "status",
                         "text": "Audio en directo no disponible (falta pyaudiowpatch)"})
            return
        p = pyaudio.PyAudio()
        try:
            self._open_captures(p, pyaudio)
            if self.cap_system is None and self.cap_mic is None:
                self.s.emit({"type": "status",
                             "text": "No encuentro ningun dispositivo de audio"})
                return
            sources = []
            if self.cap_system is not None:
                sources.append("sistema")
            if self.cap_mic is not None:
                sources.append("microfono")
            print(f"[live] capturas abiertas: {sources}", flush=True)
            self.s.emit({"type": "status",
                         "text": f"Escuchando en modo auto: {' + '.join(sources)}"})

            n_analyze = 0
            last_analysis = 0.0
            last_recognize = time.monotonic() - self.recognize_every + 14.0
            while not self.s.stop_flag.is_set():
                time.sleep(0.25)
                now = time.monotonic()
                if now - last_analysis >= self.analyze_every:
                    last_analysis = now
                    n_analyze += 1
                    if n_analyze <= 5 or n_analyze % 15 == 0:
                        print(f"[live] analyze #{n_analyze}", flush=True)
                    self._analyze()
                if now - last_recognize >= self.recognize_every:
                    last_recognize = now
                    self._recognize()
        except Exception:
            print("[live] HILO DE AUDIO MUERTO:", flush=True)
            traceback.print_exc()
            self.s.emit({"type": "status", "text": "Audio en directo detenido por un error"})
        finally:
            for cap in (self.cap_system, self.cap_mic):
                if cap is not None:
                    cap.close()
            try:
                p.terminate()
            except Exception:
                pass

    def _open_captures(self, p, pyaudio):
        try:  # loopback del dispositivo de salida por defecto
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            spk = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if not spk.get("isLoopbackDevice"):
                for lb in p.get_loopback_device_info_generator():
                    if spk["name"] in lb["name"]:
                        spk = lb
                        break
            if spk.get("isLoopbackDevice"):
                self.cap_system = _Capture(p, pyaudio, spk)
        except Exception:
            self.cap_system = None
        try:  # microfono por defecto (oye la sala)
            mic = p.get_default_input_device_info()
            self.cap_mic = _Capture(p, pyaudio, mic)
        except Exception:
            self.cap_mic = None

    def _active(self):
        """(captura, nombre) de la fuente con senal; sistema tiene prioridad."""
        if self.cap_system is not None and self.cap_system.rms() >= SILENCE_SYSTEM:
            return self.cap_system, "sistema"
        if self.cap_mic is not None and self.cap_mic.rms() >= SILENCE_MIC:
            return self.cap_mic, "microfono (musica de la sala)"
        return None, None

    # -------------------------------------------------------------- analisis
    def _analyze(self):
        seg = self.current if self.current is not None else {"energies": self.unknown_energies}
        seg_energies = seg["energies"] if "energies" in seg else self.unknown_energies
        seg_energies.append(float(getattr(self.s, "current_energy", 0.0)))

        t_now = round(float(getattr(self.s, "current_t", 0.0)), 2)
        cap, source = self._active()
        if cap is None:
            self.s.emit({"type": "audio_live", "bpm": None, "level": 0.0, "t": t_now})
            return
        if source != self._source_name:
            self._source_name = source
            self.s.emit({"type": "status", "text": f"Fuente de audio: {source}"})

        y = cap.window(8.0)
        if y is None or len(y) < cap.sr:
            self.s.emit({"type": "audio_live", "bpm": None, "level": 0.0, "t": t_now})
            return
        rms = float(np.sqrt(np.mean(y ** 2)))
        self.level_hist.append(rms)
        # Absolute level in dBFS for the dashboard meter
        self.s.live_db = 20.0 * np.log10(max(rms, 1e-6))
        ref = max(float(np.percentile(self.level_hist, 98)), 1e-6)
        level = float(np.clip(rms / ref, 0.0, 1.0))
        bpm = None
        try:
            import librosa
            onset = librosa.onset.onset_strength(y=y, sr=cap.sr)
            try:
                from librosa.feature.rhythm import tempo as tempo_fn
            except ImportError:
                tempo_fn = librosa.beat.tempo
            bpm = float(np.atleast_1d(tempo_fn(onset_envelope=onset, sr=cap.sr))[0])
            self._feed_groove(onset, cap.sr)
        except Exception:
            pass
        self.s.emit({"type": "audio_live",
                     "bpm": round(bpm, 1) if bpm else None,
                     "level": round(level, 2), "t": t_now})
        self._np_counter += 1
        if self._np_counter % 3 == 0:
            self._emit_now_playing()
            self._emit_genres()
        self._mid_track_check()

    def _feed_groove(self, onset, sr):
        """Anade al Groove Sync los ultimos ~2 s de envolvente ritmica a 10 Hz."""
        groove = getattr(self.s, "groove", None)
        if groove is None or len(onset) < 20:
            return
        frame_rate = sr / 512.0
        n_new = int(self.analyze_every * frame_rate)
        tail = onset[-n_new:] if len(onset) > n_new else onset
        self._env_ref.append(float(np.percentile(tail, 95)))
        ref = max(float(np.percentile(self._env_ref, 90)), 1e-6)
        step = max(1, int(frame_rate / 10.0))
        samples = tail[::step]
        t_end = float(getattr(self.s, "current_t", 0.0))
        n = len(samples)
        for i, v in enumerate(samples):
            t = t_end - (n - 1 - i) / 10.0
            groove.add_audio(t, float(np.clip(v / ref, 0.0, 1.5)))

    # ------------------------------------------------------- reconocimiento
    def _recognize(self):
        cap, source = self._active()
        if cap is None:
            print("[reco] sin fuente activa (silencio)", flush=True)
            return
        y = cap.window(12.0)
        if y is None or len(y) < cap.sr * 8:
            print(f"[reco] buffer insuficiente ({0 if y is None else len(y)/cap.sr:.1f}s)", flush=True)
            return
        print(f"[reco] intento con {source}: {len(y)/cap.sr:.1f}s "
              f"rms={float(np.sqrt(np.mean(y**2))):.4f}", flush=True)
        wav = os.path.join(tempfile.gettempdir(), "kadenz_live.wav")
        try:
            import soundfile as sf
            sf.write(wav, y, cap.sr, subtype="PCM_16")  # Shazam espera PCM16, no float
            from shazamio import Shazam

            async def go():
                sh = Shazam()
                try:
                    return await sh.recognize(wav)
                except AttributeError:
                    return await sh.recognize_song(wav)

            out = asyncio.run(go())
            self._reco_fails = 0
            track = (out or {}).get("track") or {}
            title = track.get("title")
            print(f"[reco] resultado: {title or 'SIN MATCH'}", flush=True)
            if not title:
                self._nomatch += 1
                if self._nomatch == 2:
                    self.s.emit({"type": "status",
                                 "text": "Suena musica pero no identifico el tema aun (¿remix o directo?)"})
                return
            self._nomatch = 0
            if self.current is not None and self.current["title"] == title:
                return
            artist = track.get("subtitle", "?")
            genre = (track.get("genres") or {}).get("primary", "?")
            self._on_track_change(title, artist, genre)
        except Exception as e:
            print(f"[reco] ERROR {type(e).__name__}: {e}", flush=True)
            self._reco_fails += 1
            if self._reco_fails >= 3 and not self._reco_warned:
                self._reco_warned = True
                self.s.emit({"type": "status",
                             "text": "Reconocimiento de canciones no disponible (¿sin red?)"})
        finally:
            if os.path.exists(wav):
                try:
                    os.remove(wav)
                except OSError:
                    pass

    # --------------------------------------------------------------- copilot
    @staticmethod
    def _mean_energy(track):
        energies = track.get("energies") or []
        return float(np.mean(energies)) if energies else None

    def _emit_now_playing(self):
        cur = self.current
        if cur is None:
            return
        scored = [t for t in self.tracks if self._mean_energy(t) is not None]
        best = max(scored, key=self._mean_energy) if scored else None
        self.s.emit({
            "type": "now_playing",
            "title": cur["title"], "artist": cur["artist"], "genre": cur["genre"],
            "avg_e": round(float(np.mean(cur["energies"])), 1) if cur["energies"] else None,
            "best": ({"title": best["title"], "avg_e": round(self._mean_energy(best), 1)}
                     if best else None),
        })

    def _emit_genres(self):
        """Score por estilo musical: energia visual media mientras sono cada genero."""
        agg = {}
        candidates = self.tracks + ([self.current] if self.current is not None else [])
        for tr in candidates:
            genre = tr.get("genre") or "?"
            energies = tr.get("energies") or []
            if len(energies) < 5:
                continue
            entry = agg.setdefault(genre, {"sum": 0.0, "n": 0, "titles": set()})
            entry["sum"] += float(np.sum(energies))
            entry["n"] += len(energies)
            entry["titles"].add(tr["title"])
        items = [{"genre": g, "score": round(v["sum"] / v["n"], 1), "tracks": len(v["titles"])}
                 for g, v in agg.items() if v["n"] > 0]
        items.sort(key=lambda x: -x["score"])
        if items:
            self.s.emit({"type": "genres", "items": items[:5]})

    def _on_track_change(self, title, artist, genre):
        t_now = float(getattr(self.s, "current_t", 0.0))
        prev = self.current
        if prev is not None:
            self.tracks.append(prev)
        self.current = {"title": title, "artist": artist, "genre": genre,
                        "start": t_now, "energies": []}
        self.s.emit({"type": "event", "kind": "track", "t": t_now,
                     "text": f"Ahora suena: {artist} – {title} ({genre})"})
        self._emit_now_playing()

        if prev is not None and len(self.tracks) >= 2:
            before = self._mean_energy(self.tracks[-2])
            after = self._mean_energy(self.tracks[-1])
            if before and after:
                delta = 100.0 * (after - before) / max(before, 1e-6)
                if delta <= -15:
                    scored = [t for t in self.tracks if self._mean_energy(t)]
                    best = max(scored, key=self._mean_energy) if scored else None
                    hint = (f" El que mejor funciono esta noche: "
                            f"«{best['title']}» ({best['genre']})." if best else "")
                    self.s.emit({"type": "event", "kind": "reco", "t": t_now,
                                 "text": (f"«{prev['title']}» bajo el ambiente un "
                                          f"{abs(delta):.0f}% respecto al tema anterior.{hint}")})
                elif delta >= 15:
                    self.s.emit({"type": "event", "kind": "reco", "t": t_now,
                                 "text": (f"«{prev['title']}» funciono: energia "
                                          f"+{delta:.0f}% respecto al anterior. "
                                          f"Ese es el camino ({prev['genre']}).")})

    def _mid_track_check(self):
        cur = self.current
        if cur is None or cur["title"] in self._mid_advised:
            return
        t_now = float(getattr(self.s, "current_t", 0.0))
        if t_now - cur["start"] < 45 or len(cur["energies"]) < 20:
            return
        all_energies = list(self.unknown_energies)
        for t in self.tracks:
            all_energies.extend(t.get("energies") or [])
        if len(all_energies) < 30:
            return
        cur_mean = float(np.mean(cur["energies"]))
        session_mean = float(np.mean(all_energies))
        if cur_mean < session_mean * 0.8:
            self._mid_advised.add(cur["title"])
            drop_pct = 100.0 * (1.0 - cur_mean / max(session_mean, 1e-6))
            self.s.emit({"type": "event", "kind": "reco", "t": t_now,
                         "text": (f"El ambiente lleva un {drop_pct:.0f}% por debajo de la "
                                  f"media de la noche con «{cur['title']}» — "
                                  f"considera cambiar de rollo.")})
