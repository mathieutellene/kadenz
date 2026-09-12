"""Input normalisation.

Decoding 4K at 30 fps starves the analysis thread, so any clip whose long side
exceeds `max_side` is transcoded ONCE to `target_side` (fast H.264, audio
untouched) and cached under data/uploads/.cache. Later runs of the same file
start instantly.
"""
import subprocess
from pathlib import Path

import cv2


def ensure_analyzable(path_str, max_side=1440, target_side=960, on_status=None):
    path = Path(path_str)
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w == 0 or max(w, h) <= max_side:
        return str(path)

    cache_dir = path.parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    out = cache_dir / f"{path.stem}_{path.stat().st_size}_{target_side}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return str(out)

    if on_status:
        on_status(f"Optimizando video {w}x{h} para analisis fluido (solo la primera vez)...")
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        vf = (f"scale='if(gte(iw,ih),{target_side},-2)'"
              f":'if(gte(iw,ih),-2,{target_side})'")
        cmd = [ffmpeg, "-y", "-i", str(path), "-vf", vf,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-c:a", "copy", str(out)]
        result = subprocess.run(cmd, capture_output=True, timeout=1800)
        if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return str(out)
    except Exception:
        pass
    return str(path)  # si falla, se analiza el original (lento pero funcional)
