"""Headless engine check -- no server, no browser.

    python -m engine.selftest data/clip.mp4 30

Prints the hello/metrics/event stream and writes an annotated frame to
data/selftest_frame.jpg for visual inspection.
"""
import queue
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.session import AnalysisSession  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m engine.selftest <video> [segundos]")
        sys.exit(1)
    video = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    q = queue.Queue(maxsize=1000)
    sess = AnalysisSession(video, cfg, q, name=Path(video).name)
    sess.start()

    t0 = time.time()
    frames = metrics = events = 0
    saved = False
    last_ms = -1
    last_snap = None
    while time.time() - t0 < budget:
        lf = sess.latest_frame
        if lf is not None and lf[0] != last_ms:
            last_ms = lf[0]
            frames += 1
            if frames == 90 and not saved:
                out = Path("data/selftest_frame.jpg")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(lf[1])
                saved = True
        try:
            msg = q.get(timeout=0.03)
        except queue.Empty:
            if not sess.is_alive():
                break
            continue
        kind = msg["type"]
        if kind == "hello":
            print("HELLO video:", msg["video"])
            print("HELLO audio:", "si" if msg.get("audio") else "no",
                  f"(bpm={msg['audio']['bpm']}, drops={msg['audio']['drops']})" if msg.get("audio") else "")
        elif kind == "status":
            print("STATUS:", msg["text"])
        elif kind == "audio":
            print(f"AUDIO listo: bpm={msg['bpm']} drops={msg['drops']}")
        elif kind == "now_playing":
            print(f"NOW PLAYING: {msg['artist']} - {msg['title']} ({msg['genre']})")
        elif kind == "metrics":
            metrics += 1
            last_snap = msg
            if metrics % 10 == 0:
                print(f"  t={msg['t']:7.1f}s energy={msg['energy']:5.1f} "
                      f"people={msg['occupancy']:3d} moving={msg['participation']:5.1f}% "
                      f"groove={msg.get('groove')} proc_fps={msg.get('proc_fps', '?')}")
        elif kind == "event":
            events += 1
            print(f"EVENT t={msg['t']}s: {msg['text']}")
        elif kind == "end":
            print("END:", msg["summary"])
            break
        elif kind == "error":
            print("ERROR:", msg["text"])
            break
    sess.stop()
    print(f"\nresultado: frames={frames} metrics={metrics} events={events}")
    print("ultimo snapshot:", last_snap)
    print("frame anotado guardado:", saved)


if __name__ == "__main__":
    main()
