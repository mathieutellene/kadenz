"""Encode annotated engine frames into a GIF via ffmpeg's two-pass palette.

PIL's quantiser gives ~7 MB on dense club footage; ffmpeg's palettegen +
paletteuse with a Bayer dither lands the same clip around 2 MB at the same
apparent quality, because it builds one optimal palette for the whole clip and
then only encodes what actually changed between frames.

    python gifenc.py <clip> <out.gif> [--fps 12] [--width 640] [--colors 80]
"""
import argparse
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_media import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=100.0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--colors", type=int, default=80)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--every", type=int, default=2)
    args = ap.parse_args()

    frames = run(args.clip, args.seconds, args.imgsz, 960)
    first = next((i for i, f in enumerate(frames) if f[2] > 0), None)
    if first is None:
        sys.exit("no frame ever had a tracked person")
    print(f"  warm-up: {first} frames before the first overlay")
    frames = frames[first + args.skip:]
    picked = frames[::args.every][:args.max_frames]
    if not picked:
        sys.exit("no frames")

    tmp = Path(tempfile.mkdtemp(prefix="kadenzgif_"))
    try:
        for i, (_, jb, _n) in enumerate(picked):
            im = Image.open(io.BytesIO(jb)).convert("RGB")
            w = args.width
            # ffmpeg's palette filters want even dimensions
            h = round(im.height * w / im.width) // 2 * 2
            im.resize((w, h), Image.LANCZOS).save(tmp / f"f{i:04d}.png")

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        vf = (f"fps={args.fps},split[a][b];"
              f"[a]palettegen=max_colors={args.colors}:stats_mode=diff[p];"
              f"[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle")
        cmd = [ff, "-y", "-framerate", str(args.fps),
               "-i", str(tmp / "f%04d.png"), "-vf", vf,
               "-loop", "0", str(args.out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-1200:])
            sys.exit("ffmpeg failed")
        mb = Path(args.out).stat().st_size / 1e6
        print(f"  wrote {args.out} ({len(picked)} frames, {mb:.1f} MB)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
