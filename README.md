# CrowdPulse

**A camera that reads the dancefloor — and tells the DJ what to play next.**

Music recommenders are open loop. Spotify knows what you played; it never knows
whether the floor emptied. CrowdPulse closes that loop with the one signal
nobody has: **how the crowd actually reacted, measured in real time by computer
vision.**

```
 track plays ──▶ vision measures the crowd ──▶ floor profile updates ──▶ next track
      ▲                                                                      │
      └──────────────────────────────────────────────────────────────────────┘
```

![CrowdPulse overlay](docs/media/demo.gif)

Everything runs locally on a laptop CPU. No GPU, no cloud, no per-frame API calls.

---

## The dashboard

![CrowdPulse dashboard](docs/media/dashboard.png)

Left: the footage with hairline pose overlays and a picture-in-picture of what
the model actually sees. Right: the live read of the room, the track playing,
and the next-track recommendation **with the reason it was chosen**.

---

## What it measures

| Metric | What it answers | How |
|---|---|---|
| **Crowd energy** | How hard is the room going? | Dense optical flow inside person boxes, ranked against the session's own history (auto-calibrating, 0–100) |
| **People** | How many are on the floor? | YOLOv8-pose detection + ByteTrack identities |
| **Moving %** | Are they dancing or just standing? | Share of tracked people whose local motion clears a threshold |
| **Floor flux** | Is the floor filling or emptying? | Confirmed track entries minus exits per minute |
| **Groove sync** ⭐ | Do they move *with* the music, or just move? | Rolling Pearson correlation between motion energy and the audio onset envelope, searched over ±1.5 s of lag |
| **Level (dB SPL)** | How loud is the room? | RMS mapped to an SPL estimate (a laptop mic is not a calibrated meter — see *What is real*) |
| **Per-person energy** | Who is actually going off? | Same flow signal restricted to each person's box, percentile-ranked across the session |

Colour means the same thing everywhere — overlays, meters and charts share one
ramp: **cyan = still → amber → hot pink = going off.**

![Per-person overlay](docs/media/overlay.jpg)

*Cyan skeletons are spectators standing still; the pink ones in the middle are
the people actually dancing. Nobody labelled that — it falls out of the motion
measurement.*

---

## The closed loop

`engine/dj.py` is a contextual-bandit-style recommender. Every track that plays
leaves a reward: the mean crowd energy it produced, measured by the vision
pipeline. Genres and 5-BPM tempo bands accumulate those rewards, and the next
pick maximises expected response — constrained by what a DJ can actually
beatmatch (±6% BPM) and with a small exploration term so the set does not
collapse into one genre.

```
played  Melodic Techno  124bpm -> response 55   delta   n/a
played  Peak Techno     138bpm -> response 88   delta  +61%
played  Ambient House   112bpm -> response 23   delta  -68%
played  Peak Techno     136bpm -> response 88   delta  +60%

NEXT TRACK (closed-loop recommendation):
  Iron Garden — Bruk Theory   Peak Techno  134bpm  [mixable]
     -> Peak Techno running +46% above tonight's average
```

The engine learned this floor's taste in four tracks, and it explains itself.

---

## How it works

Three loops run at their own pace so the video never waits for the model:

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

**Detection size is the main quality/speed dial** (`det_imgsz` in `config.yaml`).
On the demo clip, measured on a laptop CPU with no GPU:

| `det_imgsz` | People found | Inference |
|---|---|---|
| 640 | ~15 | ~90 ms |
| **960 (default)** | **~20** | ~180 ms |
| 1280 (crowd mode) | ~27–34 | ~240 ms |

The stills in this README use crowd mode. Because display and analysis run on
separate threads, raising it costs detection latency, never video smoothness.

Headless check, no browser:

```bash
python -m engine.selftest path/to/clip.mp4 30
```

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
| Groove sync, BPM, drops, track ID (Shazam) | **Real when there is real audio.** Stock footage ships silent audio tracks, so the demo shows `n/a` rather than a number |
| Track feed, genre and the recommendations built on them | **Simulated** — a synthetic catalogue of fictional tracks with realistic audio features (`engine/dj.py`). The *crowd response driving the recommender is measured for real*; only the music metadata is stand-in, because licensed catalogue APIs are not publicly available |
| Level in dB SPL | **Estimated** — a laptop mic is not a calibrated SPL meter; in simulated mode the level follows the simulated track |

Simulated values are tagged `SIMULATED` / `· sim` in amber in the UI, never
presented as measurements. Silent audio is rejected outright: analysing it
would produce a fabricated BPM.

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
- Person IDs are anonymous integers that die with the session. No faces are
  stored, no identity is computed, only aggregates leave the engine.

---

## License

**AGPL-3.0** — required, because this depends on Ultralytics YOLO, which is
AGPL-3.0. Swapping the detector for an Apache-2.0 model (RT-DETR, D-FINE) would
allow a permissive licence.

Demo footage: [Pexels](https://www.pexels.com/) free licence. The synthetic
track catalogue is fictional.
