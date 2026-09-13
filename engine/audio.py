"""Offline audio features for a video file: BPM, level, onset envelope, drops.

Computed once per session in a background thread so the video never waits. If
ffmpeg/librosa are missing, the file has no audio, or the track is silent, this
returns None and the system keeps running on vision alone -- analysing silence
would fabricate a BPM.

Also: one-shot track identification via Shazam.
"""
import asyncio
import os
import subprocess
import tempfile

import numpy as np

try:  # shazamio decodifica con pydub: apuntarlo al ffmpeg embebido
    import pydub
    import imageio_ffmpeg
    pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass


def extract_audio_features(video_path, target_rate=10):
    try:
        import imageio_ffmpeg
        import librosa
    except Exception:
        return None
    wav = os.path.join(tempfile.gettempdir(), "kadenz_audio.wav")
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "22050", wav]
        subprocess.run(cmd, capture_output=True, timeout=300, check=True)
        y, sr = librosa.load(wav, sr=22050, mono=True)
        if y.size < sr:
            return None
        # Stock footage often ships a silent audio track. Analysing it would
        # produce a fabricated BPM, so treat near-silence as "no audio".
        if 20.0 * np.log10(max(float(np.sqrt(np.mean(y ** 2))), 1e-9)) < -55.0:
            return None
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        try:
            from librosa.feature.rhythm import tempo as tempo_fn
        except ImportError:
            tempo_fn = librosa.beat.tempo
        bpm = float(np.atleast_1d(tempo_fn(onset_envelope=onset, sr=sr, hop_length=hop))[0])

        frame_rate = sr / hop
        n_out = int(len(rms) / frame_rate * target_rate)
        idx = (np.arange(max(n_out, 1)) * frame_rate / target_rate).astype(int)
        idx = idx[idx < len(rms)]
        rms_t = rms[idx]
        rms_norm = np.clip(rms_t / (np.percentile(rms_t, 98) + 1e-9), 0.0, 1.0)
        # True level in dBFS (0 dBFS = full scale). Club PA levels are ~-12 dBFS.
        db = 20.0 * np.log10(np.maximum(rms_t, 1e-6))
        onset_t = onset[idx[idx < len(onset)]]
        onset_norm = np.clip(onset_t / (np.percentile(onset_t, 98) + 1e-9), 0.0, 1.0)
        return {
            "rate": target_rate,
            "rms": [round(float(v), 3) for v in rms_norm],
            "db": [round(float(v), 1) for v in db],
            "onset": [round(float(v), 3) for v in onset_norm],
            "bpm": round(bpm, 1),
            "drops": _detect_drops(rms_norm, target_rate),
        }
    except Exception:
        return None
    finally:
        if os.path.exists(wav):
            try:
                os.remove(wav)
            except OSError:
                pass


def recognize_file_segment(video_path, at_fraction=0.5, seconds=12):
    """Identify the track playing in a segment of the file (Shazam)."""
    try:
        import cv2
        import imageio_ffmpeg
        from shazamio import Shazam
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        duration = frames / fps if fps else 0
        start = max(0.0, duration * at_fraction - seconds / 2)
        wav = os.path.join(tempfile.gettempdir(), "kadenz_reco.wav")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(start),
                        "-i", str(video_path), "-t", str(seconds),
                        "-ac", "1", "-ar", "44100", wav],
                       capture_output=True, timeout=120, check=True)

        async def go():
            sh = Shazam()
            try:
                return await sh.recognize(wav)
            except AttributeError:
                return await sh.recognize_song(wav)

        out = asyncio.run(go())
        os.remove(wav)
        track = (out or {}).get("track") or {}
        if not track.get("title"):
            return None
        return {
            "title": track.get("title"),
            "artist": track.get("subtitle", "?"),
            "genre": (track.get("genres") or {}).get("primary", "?"),
        }
    except Exception:
        return None


def _detect_drops(rms, rate):
    """v1 heuristic: a sustained jump in level after a quiet build-up."""
    drops = []
    w = rate * 4
    i = w
    mean = float(np.mean(rms))
    while i < len(rms) - rate:
        pre = rms[i - w:i]
        post = rms[i:i + rate]
        if pre.mean() < mean * 0.85 and post.mean() > mean * 1.15 and post.max() > pre.mean() * 1.6:
            drops.append(round(i / rate, 1))
            i += rate * 8  # refractario: un drop no puede repetirse en 8 s
        else:
            i += max(rate // 2, 1)
    return drops[:20]
