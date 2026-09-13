"""Render annotated media straight out of the engine.

Runs a real AnalysisSession headlessly and collects the annotated JPEG frames it
publishes, so whatever ends up in the README is literally what the app draws --
no separate rendering path that could drift from the product.

    python gen_media.py <clip> <out.gif|out.jpg> [--seconds N] [--imgsz 1280]
"""
import argparse
import io
import queue
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.session import AnalysisSession  # noqa: E402


def run(clip, seconds, imgsz, width):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["model"]["det_imgsz"] = imgsz
    cfg["video"]["display_width"] = width
    q = queue.Queue(maxsize=4000)
    sess = AnalysisSession(str(clip), cfg, q, name=Path(clip).name)
    sess.start()

    frames, last_ms, peak = [], -1, {"people": 0, "energy": 0.0}
    t0 = time.time()
    while time.time() - t0 < seconds:
        lf = sess.latest_frame
        if lf is not None and lf[0] != last_ms:
            last_ms = lf[0]
            # Record how many skeletons were on this frame. The display thread
            # starts emitting before the first inference lands, so the opening
            # frames carry no overlay at all -- a fixed --skip cannot know how
            # many that is, and it varies with model load time.
            frames.append((lf[0], lf[1], len(sess._track_state)))
        try:
            m = q.get(timeout=0.02)
        except queue.Empty:
            if not sess.is_alive():
                break
            continue
        if m["type"] == "metrics":
            peak["people"] = max(peak["people"], m["occupancy"])
            peak["energy"] = max(peak["energy"], m["energy"])
        elif m["type"] in ("end", "error"):
            break
    sess.stop()
    print(f"  {len(frames)} frames, peak {peak['people']} people, "
          f"peak energy {peak['energy']:.0f}")
    return frames


def _sharpness(jpeg):
    """Variance of the Laplacian: cheap proxy for 'is this frame crisp'."""
    im = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"), dtype=float)
    gy, gx = np.gradient(im)
    return float((gx ** 2 + gy ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=75.0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--skip", type=int, default=4,
                    help="extra frames to drop after the overlay first appears")
    ap.add_argument("--gif-frames", type=int, default=60)
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--duration", type=int, default=80)
    args = ap.parse_args()

    print(f"rendering {Path(args.clip).name} -> {args.out}")
    frames = run(args.clip, args.seconds, args.imgsz, args.width)
    first = next((i for i, f in enumerate(frames) if f[2] > 0), None)
    if first is None:
        sys.exit("no frame ever had a tracked person")
    frames = frames[first + args.skip:]
    if not frames:
        sys.exit("no frames after the warm-up")
    print(f"  warm-up: {first} frames before the first overlay")

    out = Path(args.out)
    if out.suffix.lower() == ".gif":
        picked = frames[::args.every][:args.gif_frames]
        imgs = []
        for _, jb, _n in picked:
            im = Image.open(io.BytesIO(jb)).convert("RGB")
            w = args.gif_width
            imgs.append(im.resize((w, round(im.height * w / im.width)), Image.LANCZOS))
        # One palette for the whole clip: per-frame palettes make every frame a
        # full keyframe and blow the file up. Dithering costs a lot of bytes on
        # noisy club footage for no visible gain at this size, so skip it.
        pal = imgs[0].quantize(colors=args.colors, method=Image.MEDIANCUT)
        imgs = [im.quantize(palette=pal, dither=Image.NONE) for im in imgs]
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     duration=args.duration, loop=0, optimize=True)
        print(f"  wrote {out} ({len(imgs)} frames, {out.stat().st_size/1e6:.1f} MB)")
    else:
        # Pick the crispest frame in the second half: the tracker has settled and
        # the energy ramp has had time to spread across the ramp's colours.
        tail = frames[len(frames) // 2:]
        best = max(tail, key=lambda f: _sharpness(f[1]))
        Image.open(io.BytesIO(best[1])).convert("RGB").save(out, quality=92)
        print(f"  wrote {out} ({out.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
