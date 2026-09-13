# Demo footage

The three clips the README's images are made from, kept in the repo so the
media can be regenerated after any change to the overlay, the colour ramp or
the dashboard — not just re-downloaded and hoped to be the same file.

All three are stock clips under the [Pixabay content
licence](https://pixabay.com/service/license-summary/), which permits use and
redistribution as part of a work. The original filenames carried the stock ID,
recorded here so each can be traced back or replaced.

| file | original filename | scene | used for |
|---|---|---|---|
| `concert-open-air.mp4` | `12768106-hd_1920_1080_30fps.mp4` | elevated view of an open-air concert | `docs/media/dashboard.png` |
| `night-stage.mp4` | `12248654_1920_1440_30fps.mp4` | outdoor night stage, people dancing and sitting | `docs/media/demo.gif` |
| `rooftop-club.mp4` | `12158875_2562_1440_32fps.mp4` | packed rooftop bar, dense distant crowd | `docs/media/overlay.jpg` |

## Why these three and not others

Seven clips were screened by running the detector over sampled frames. What
matters for a demo image is not how many people are in shot but how many the
model can actually resolve — a frame full of untracked people reads as a broken
system:

| clip | detected | keypoints | covers the people in frame? |
|---|---|---|---|
| **night-stage** | 24.2 | **13.2/17** | yes, nearly everyone |
| rooftop-club | 20.1 | 12.2/17 | no — hundreds left untracked |
| vertical (`15668967`, not kept) | 36.5 | 13.0/17 | yes, but 9:16 is unusable as a banner |
| `14445443`, `112924`, `19300221` (not kept) | 6–23 | 7.3–11.7 | no |

`concert-open-air.mp4` is the only clip of the seven with a **real audio track**
(≈123 BPM). That is why the dashboard screenshot is taken on it: its BPM, dB SPL
and groove sync are measured rather than blank.

`rooftop-club.mp4` is kept deliberately as the counter-example. It is the frame
that shows the density limit — the front rows tracked, the crowd behind them
invisible — and the README uses it to say so.

## Regenerating the media

```bash
# hero animation (night stage)
python scripts/gifenc.py assets/footage/night-stage.mp4 docs/media/demo.gif \
    --width 520 --colors 200 --max-frames 40 --fps 10

# per-person still (rooftop)
python scripts/gen_media.py assets/footage/rooftop-club.mp4 docs/media/overlay.jpg \
    --seconds 90 --width 1440
```

Both scripts run a real `AnalysisSession`, so whatever lands in the README is
literally what the app draws. They also record the track count per frame and
start at the first frame that actually carries an overlay — the display thread
emits before the first inference lands, and a fixed frame skip cannot know how
many of those there will be.

`docs/media/dashboard.png` is a screenshot of the live dashboard rather than a
render, taken on a looped build of the concert clip so a set has time to
develop:

```bash
ffmpeg -stream_loop 27 -i assets/footage/concert-open-air.mp4 -t 180 \
    -vf scale=1280:-2 -c:v libx264 -preset veryfast -crf 24 \
    -c:a aac -b:a 128k data/demo_long.mp4
```

Then `python run.py`, load that file, and capture once the set has built up
enough history to show a divergence.
