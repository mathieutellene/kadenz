"""Open loop vs closed loop, on the same catalogue.

Runs a ten-track set against a synthetic floor with a definite taste -- a room
that wants tech house and does not want pop -- and prints what each model
wanted to play. No camera, no video, no model weights -- this
exercises `engine/dj.py` alone, so the recommendation argument can be checked in
two seconds:

    python scripts/sim_loop.py

The open-loop column is a content-based recommender (audio-feature similarity +
popularity prior). The closed-loop column is the same catalogue re-ranked by the
crowd response the vision pipeline would have measured. Rows marked with a
diamond are the ones where the crowd overruled the feature match.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.dj import DJEngine   # noqa: E402

# How this fictional floor reacts, as mean crowd energy 0-100 per genre. In the
# real system this number is not invented: it is measured by optical flow.
FLOOR = {"Tech House": 88, "French House": 76, "Future House": 70,
         "Deep House": 62, "Big Room": 48, "Indie Dance": 44,
         "Synth Pop": 28, "Synthwave": 22}
TRACKS = 10
SPREAD = 6.0      # how noisy a single track's response is


def main(seed=7, floor_seed=3):
    dj = DJEngine(seed=seed)
    rng = random.Random(floor_seed)
    t = 0.0

    print(" #  CLOSED LOOP PLAYED                              resp      Δ   "
          "OPEN LOOP WANTED INSTEAD")
    print("─" * 99)

    for i in range(TRACKS):
        cmp_ = dj.loop_compare(1, count=True)
        openl, closed = cmp_["open"][0], cmp_["closed"][0]
        dj.start_track(closed, t, open_pick={"title": openl["title"],
                                             "genre": openl["genre"]})
        # The floor reacts to whatever is on the deck.
        for _ in range(20):
            dj.sample(max(0.0, min(100.0, rng.gauss(FLOOR[closed["genre"]], SPREAD))))
        t += 14
        done = dj.close_current(t)

        delta = "" if done["delta"] is None else "{:+.0f}%".format(done["delta"])
        mark = "" if cmp_["agree"] else "◆"
        played = "{} — {}".format(closed["title"], closed["artist"])
        wanted = openl["title"] if cmp_["agree"] else "{} {}".format(openl["title"], mark)
        print("{:>2}  {:<47} {:>4.0f}  {:>6}   {}".format(
            i + 1, played[:47], done["response"], delta, wanted))

    print("─" * 99)
    fin = dj.loop_compare(3)
    print("divergence: {}% — {} of {} picks corrected by the crowd".format(
        fin["divergence_pct"], fin["corrected"], fin["compared"]))
    print("final call: {}".format(
        fin["note"].replace("Crowd response overrides the feature match: ", "")))

    print("\nwhat the floor taught it, from nothing, in {} tracks:".format(TRACKS))
    rows = dj.genre_scores()
    for a, b in zip(rows[::2], list(rows[1::2]) + [None]):
        left = "  {:<16} {:>5.1f}".format(a["genre"], a["score"])
        right = "" if b is None else "     {:<16} {:>5.1f}".format(b["genre"], b["score"])
        print(left + right)


if __name__ == "__main__":
    main()
