"""Emit the browser demo's catalogue from engine/dj.py.

The live page re-implements the recommender in JavaScript, which means the
seventeen tracks and their audio features would exist twice and drift apart the
first time anyone edits one of them. They don't: this writes docs/catalogue.js
straight out of the Python, and `python scripts/build_catalogue.py --check`
fails if the committed file has fallen behind (run it after touching CATALOGUE).

    python scripts/build_catalogue.py            # regenerate
    python scripts/build_catalogue.py --check    # verify, exit 1 if stale
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.dj import CATALOGUE, MIXABLE_BPM_PCT, _BPM_HI, _BPM_LO, _SIM_W  # noqa: E402

OUT = ROOT / "docs" / "catalogue.js"
HEADER = """// GENERATED FILE -- do not edit by hand.
// Source of truth is engine/dj.py; regenerate with:
//     python scripts/build_catalogue.py
"""


def render():
    body = {
        "CATALOGUE": CATALOGUE,
        "MIXABLE_BPM_PCT": MIXABLE_BPM_PCT,
        "SIM_W": _SIM_W,
        "BPM_LO": _BPM_LO,
        "BPM_HI": _BPM_HI,
    }
    lines = [HEADER]
    lines.append("export const CATALOGUE = " + json.dumps(body["CATALOGUE"], indent=2) + ";\n")
    lines.append(f"export const MIXABLE_BPM_PCT = {body['MIXABLE_BPM_PCT']};")
    lines.append("export const SIM_W = " + json.dumps(body["SIM_W"]) + ";")
    lines.append(f"export const BPM_LO = {body['BPM_LO']};")
    lines.append(f"export const BPM_HI = {body['BPM_HI']};\n")
    return "\n".join(lines)


def main():
    text = render()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print("STALE: docs/catalogue.js does not match engine/dj.py")
            print("  run: python scripts/build_catalogue.py")
            sys.exit(1)
        print(f"OK: docs/catalogue.js matches engine/dj.py ({len(CATALOGUE)} tracks)")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(CATALOGUE)} tracks)")


if __name__ == "__main__":
    main()
