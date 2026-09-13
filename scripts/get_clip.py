"""Descarga un clip de prueba y lo recorta para la biblioteca.

Uso:
  .venv\\Scripts\\python.exe scripts\\get_clip.py "ytsearch1:boiler room set" salida.mp4 --start 600 --dur 90
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="URL de YouTube o 'ytsearch1:consulta'")
    ap.add_argument("output", help="nombre del archivo destino (en data/uploads)")
    ap.add_argument("--start", type=float, default=0.0, help="segundo inicial del recorte")
    ap.add_argument("--dur", type=float, default=90.0, help="duracion del recorte en segundos")
    args = ap.parse_args()

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = Path(tempfile.gettempdir()) / "kadenz_raw.mp4"

    print("Descargando...")
    cmd = [sys.executable, "-m", "yt_dlp",
           "-f", "mp4[height<=720]/best[height<=720]/best",
           "--no-playlist", "-o", str(tmp), "--force-overwrites", args.query]
    subprocess.run(cmd, check=True)

    dest = ROOT / "data" / "uploads" / args.output
    print(f"Recortando {args.dur}s desde t={args.start}s -> {dest.name}")
    subprocess.run([ffmpeg, "-y", "-ss", str(args.start), "-i", str(tmp),
                    "-t", str(args.dur), "-c", "copy", str(dest)], check=True)
    print("Listo:", dest)


if __name__ == "__main__":
    main()
