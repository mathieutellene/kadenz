# Kadenz `v1.0`

**A camera that reads the dancefloor — and closes the loop a music recommender can't.**

> First public version. The vision pipeline and the closed-loop recommender both
> work end to end; the roadmap below is what v1.1 is for.

![Kadenz overlay](docs/media/demo.gif)

<sub>Night stage. The dancers run amber; the people sitting on the steps in the foreground stay cool. Nothing is labelled by hand — the colour *is* the measured motion.</sub>

---

## The problem: recommendation is an open loop

<img src="docs/media/spotify-mark.svg" height="15" alt="Spotify"> **Spotify's**
recommender is extremely good at one question — *given this track, what sounds
like it?* It answers from audio features (`energy`, `danceability`, `valence`,
`tempo`, `key`) plus what millions of other listeners did next. It is the best
open-loop music model ever built.

But it is **open loop**. It knows what was played. It never finds out whether
the floor emptied.

A DJ closes that loop by eye, every twenty seconds, all night. **Kadenz closes
it with computer vision** — the same catalogue, the same audio features, but
re-ranked by the one signal nobody has: *how the crowd actually reacted,
measured in real time.*

```
       ┌──────────────────────────── the loop ────────────────────────────┐
       │                                                                  │
       ▼                                                                  │
  ┌─────────┐      ┌────────┐      ┌──────────┐      ┌───────────────┐    │
  │ SPOTIFY │─────▶│  DECK  │─────▶│  VISION  │─────▶│ CROWD RESPONSE│────┘
  │catalogue│      │ track  │      │ YOLO-pose│      │  reward 0-100 │
  │ +feats  │      │ playing│      │ + optflow│      │  per track    │
  └─────────┘      └────────┘      └──────────┘      └───────────────┘
    open loop stops here ─┘                      └─ Kadenz starts here
```

<sub>Kadenz is not affiliated with or endorsed by Spotify. The catalogue in v1.0
is a synthetic stand-in with realistic audio features — see
<a href="#what-is-real-and-what-is-simulated">What is real</a>.</sub>

Everything runs locally on a laptop CPU. No GPU, no cloud, no per-frame API calls.

---

## Both models run at once, and you can watch them disagree

`engine/dj.py` computes **two rankings of the same catalogue on every track
change**, so the premise is testable instead of merely asserted:

| | ranks by | sees the room? |
|---|---|---|
| `recommend_open()` — **open loop** | audio-feature similarity to what's playing + popularity prior | ❌ |
| `recommend()` — **closed loop** | the crowd response *this floor* gave each genre and tempo band tonight | ✅ |

A ten-track simulated set against a floor that happens to love peak techno
(`scripts/sim_loop.py`):

```
 #  PLAYING                  resp      Δ   OPEN LOOP WANTS   CLOSED LOOP PICKS
──────────────────────────────────────────────────────────────────────────────
 1  Dust and Salt              66           Velvet Hours      Dust and Salt   ◆
 2  Velvet Hours               46    -30%   Velvet Hours      Velvet Hours
 3  Paloma                     63    +14%   Sunday Chrome     Paloma          ◆
 4  Halogen                    55     -5%   Sunday Chrome     Halogen         ◆
 5  Paper Lantern              44    -24%   Sunday Chrome     Paper Lantern   ◆
 6  Low Ceiling                74    +34%   Sunday Chrome     Low Ceiling     ◆
 7  Pirate Radio               72    +24%   Sunday Chrome     Pirate Radio    ◆
 8  Glass Field                23    -62%   Sunday Chrome     Glass Field     ◆
 9  Iron Garden                88    +59%   Sunday Chrome     Iron Garden     ◆
10  Afterburn                  88    +49%   Sunday Chrome     Afterburn       ◆
──────────────────────────────────────────────────────────────────────────────
divergence: 90% — 9 of 10 picks corrected by the crowd
final call: Peak Techno over Disco House (#10 on audio features alone)
```

Two things to notice, and they are the whole argument:

1. **The open loop gets stuck.** *Sunday Chrome* is the most popular track in
   the catalogue, so it wins seven picks in a row. That is not a bug in the
   baseline — it is popularity bias, the known failure mode of
   content-plus-popularity recommenders, reproduced faithfully.
2. **The closed loop explores, then commits.** It pays for information early
   (*Glass Field*, response 23, −62%), learns the room, and lands on peak techno
   at response 88. The track it finally plays ranks **#10 of 17** on audio
   features alone. No open-loop model would ever reach it.

The genre table it learned, from nothing, in ten tracks:

```
  Peak Techno       87.7     Breakbeat         72.7
  Afro House        64.6     Melodic Techno    55.1
  Disco House       45.7     Deep House        44.0
  Ambient House     22.6
```

---

## The dashboard

![Kadenz dashboard](docs/media/dashboard.png)

Left: the footage with hairline pose overlays and a picture-in-picture of what
the model actually sees. Right: the live read of the room — and the two
recommenders stacked, open loop above, closed loop below, with the crowd's
verdict between them.

---

## What it measures

| Metric | What it answers | How |
|---|---|---|
| **Crowd energy** | How hard is the room going? | Dense optical flow inside person boxes, ranked against the session's own history (auto-calibrating, 0–100). **This is the reward signal.** |
| **People** | How many are on the floor? | YOLOv8-pose detection + ByteTrack identities |
| **Moving %** | Are they dancing or just standing? | Share of tracked people whose local motion clears a threshold |
| **Floor flux** | Is the floor filling or emptying? | Confirmed track entries minus exits per minute |
| **Groove sync** ⭐ | Do they move *with* the music, or just move? | Rolling Pearson correlation between motion energy and the audio onset envelope, searched over ±1.5 s of lag |
| **Level (dB SPL)** | How loud is the room? | RMS mapped to an SPL estimate (a laptop mic is not a calibrated meter — see *What is real*) |
| **Per-person energy** | Who is actually going off? | Same flow signal restricted to each person's box, percentile-ranked across the session |

Colour means the same thing everywhere — overlays, meters and charts share one
ramp: **cyan = still → amber → hot pink = going off.**

![Per-person overlay](docs/media/overlay.jpg)

*Rooftop club, and an honest frame: the front rows are tracked, the packed crowd
behind them is not. At that distance people are a few pixels tall and a person
detector cannot see them at all (see [Limitations](#limitations)). Among the
people it does resolve, the colour spread from cyan to orange is the measured
motion — who is dancing and who is standing still, with nothing labelled by
hand.*

---

## How the loop is implemented

**The reward.** Every track holds the deck for a fixed span. Throughout, the
vision pipeline samples crowd energy at 2 Hz; on track change those samples are
averaged into a single scalar — the reward that track earned from *this* room —
and compared against the session's running mean to produce a `Δ%`.

**The policy.** Contextual-bandit style. Rewards accumulate into two tables,
genre and 5-BPM tempo band. The next pick maximises expected response:

```
score = 0.55·genre_reward + 0.30·tempo_reward + 15·danceability
        − 120·max(0, bpm_drift − 6%)      ← unmixable tracks are unusable
        − 25 if already played            ← don't repeat the set
        + U(0, 4)                         ← exploration: keep sampling
```

The `±6%` term is not a heuristic pulled from nowhere — it is the range a DJ can
beatmatch without audible artefacts. A recommender that ignores it produces
suggestions no human can actually mix.

**The explanation.** Every pick ships with the sentence that justifies it
(*"Peak Techno running +59% above tonight's average"*). A DJ will not take an
instruction from a black box mid-set, so the engine argues its case.

---

## How it runs

Three loops at their own pace, so the video never waits for the model:

```
  capture ──▶ DISPLAY THREAD   decode, scale once, draw last known state, 18 fps
                  │                      ▲
                  ▼                      │ shared state
              ANALYSIS THREAD   YOLOv8-pose + ByteTrack + Farnebäck optical flow
                  │
              AUDIO THREAD      BPM, onset envelope, drops, level (librosa)
                  │
              METRICS 2 Hz ──▶ DJ engine ──▶ WebSocket ──▶ browser
```

Frames go over the WebSocket as binary (4-byte timestamp + JPEG) on a
**latest-wins** slot — if the browser falls behind, frames are dropped rather
than queued, so playback never accumulates lag. Everything else is JSON.

**Stack:** Python · PyTorch · Ultralytics YOLOv8-pose · ByteTrack (supervision) ·
OpenCV · librosa · FastAPI + WebSockets · ECharts.

---

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/macOS: .venv/bin/pip
python run.py                                     # -> http://127.0.0.1:8765
```

Model weights (~7 MB) download on first run into `models/`. Click **START
EVENT** and pick any video of a crowd. Nothing is stored: uploads are wiped on
start and on each new file.

Watch the recommender reason on its own, no camera needed:

```bash
python scripts/sim_loop.py               # the open-vs-closed table above
python -m engine.selftest clip.mp4 30    # headless vision check
```

**Detection size is the main quality/speed dial** (`det_imgsz` in `config.yaml`).
On the demo clip, measured on a laptop CPU with no GPU:

| `det_imgsz` | People found | Inference |
|---|---|---|
| 640 | ~15 | ~90 ms |
| **960 (default)** | **~20** | ~180 ms |
| 1280 (crowd mode) | ~27–34 | ~240 ms |

The stills in this README use crowd mode. Because display and analysis run on
separate threads, raising it costs detection latency, never video smoothness.

Live camera mode exists (`{"type":"start","webcam":true}` over the WebSocket,
`webcam_index` in `config.yaml`) and captures system audio or the mic on Windows
via WASAPI loopback, which enables real BPM, real Shazam track ID and real
Groove Sync.

---

## What is real and what is simulated

Being explicit about this, because a demo that blurs the line is worthless:

| | Status |
|---|---|
| Person detection, tracking, pose, motion energy, occupancy, flux, per-person energy | **Real** — measured from the footage |
| Groove sync, BPM, drops, track ID (Shazam) | **Real when there is real audio.** Silent footage shows `n/a` rather than a number |
| **Both recommenders** — the open-loop ranking and the closed-loop re-ranking | **Real code, really executed.** Both rank the live catalogue on every track change; the divergence number is counted, not written |
| The catalogue itself — titles, artists, genres, audio features | **Simulated** — 17 fictional tracks with realistic features (`engine/dj.py`). No licensed catalogue API is publicly available for this use; swap `CATALOGUE` for a real provider's response and nothing else changes |
| Level in dB SPL | **Estimated** — a laptop mic is not a calibrated SPL meter |

The crowd response that drives the loop is **measured for real** — that is the
half that matters, and the half that does not exist anywhere else. Simulated
values are tagged `SIMULATED` / `· sim` in amber in the UI, never presented as
measurements. Silent audio is rejected outright: analysing it would produce a
fabricated BPM.

---

## Limitations

- **Drone/overhead festival footage does not work.** People 2–3 px tall are
  invisible to a person detector — that needs density estimation
  (DM-Count/P2PNet), not detection. Measured: 0 detections across four aerial
  clips at every resolution tested.
- **Dark club footage is hard.** Backlit silhouettes under strobes are the worst
  case for COCO-trained detectors; an elevated, reasonably lit angle works far
  better. Strobe frames are gated out of the flow signal by a luminance check.
- Energy is **relative to the session**, not an absolute scale: 90 tonight is
  not comparable to 90 last night.
- The reward is **aggregate energy**, not preference. It cannot separate a floor
  that loves a track from a floor that is merely warm.
- Person IDs are anonymous integers that die with the session. No faces are
  stored, no identity is computed, only aggregates leave the engine.

## Roadmap to v1.1

- A real catalogue behind the same interface — `CATALOGUE` is the only seam.
- Density estimation for aerial footage, so festivals stop being a blind spot.
- Per-section rewards: the loop scores a whole track today, not the drop.

---

## License

**AGPL-3.0** — required, because this depends on Ultralytics YOLO, which is
AGPL-3.0. Swapping the detector for an Apache-2.0 model (RT-DETR, D-FINE) would
allow a permissive licence.

Demo footage: [Pixabay](https://pixabay.com/) content licence — one scene per
visual: night stage (the animation above), rooftop club (the per-person still),
open-air concert (the dashboard). The track catalogue is fictional.
Spotify is a trademark of Spotify AB; the mark appears here to identify the
class of system Kadenz complements, and implies no affiliation or endorsement.
