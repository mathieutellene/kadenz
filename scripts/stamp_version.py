"""Stamp one cache-busting version across every asset the live page loads.

GitHub Pages serves static files with a long-lived cache and no way to set
headers, so a returning visitor keeps whatever it already has. `index.html`
carried a `?v=` on its stylesheet and entry script, but `live.js` imports
`./vision.js`, `./dj.js`, `./audio.js` and `./backdrop.js` by bare specifier --
so a change inside any of those modules was invisible to anyone who had loaded
the page before. That is exactly how a fixed noise gate sat on the server while
the browser went on running the broken one.

One version, stamped everywhere, bumped in one command:

    python scripts/stamp_version.py          # bump to the next integer
    python scripts/stamp_version.py 42       # or set it explicitly
    python scripts/stamp_version.py --check  # fail if anything is out of step
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs" / "index.html"
ENTRY = ROOT / "docs" / "live.js"
MODULES = ("dj", "vision", "audio", "backdrop", "catalogue")


def current():
    m = re.search(r'live\.css\?v=(\d+)', HTML.read_text(encoding="utf-8"))
    return int(m.group(1)) if m else 0


def versions():
    """Every version number currently stamped anywhere, for --check."""
    found = set()
    html = HTML.read_text(encoding="utf-8")
    found.update(int(v) for v in re.findall(r'live\.(?:css|js)\?v=(\d+)', html))
    for f in (ENTRY, *(ROOT / "docs" / f"{m}.js" for m in MODULES)):
        if not f.exists():
            continue
        for _, v in re.findall(r'from "\./(\w+)\.js\?v=(\d+)"', f.read_text(encoding="utf-8")):
            found.add(int(v))
    return found


def stamp(v):
    html = HTML.read_text(encoding="utf-8")
    html = re.sub(r'live\.css\?v=\d+', f"live.css?v={v}", html)
    html = re.sub(r'live\.js\?v=\d+', f"live.js?v={v}", html)
    HTML.write_text(html, encoding="utf-8", newline="\n")

    for f in (ENTRY, *(ROOT / "docs" / f"{m}.js" for m in MODULES)):
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8")
        out = re.sub(r'from "\./(\w+)\.js(?:\?v=\d+)?"', rf'from "./\1.js?v={v}"', src)
        if out != src:
            f.write_text(out, encoding="utf-8", newline="\n")


def main():
    args = [a for a in sys.argv[1:]]
    if "--check" in args:
        vs = versions()
        if len(vs) > 1:
            print(f"MISMATCHED asset versions: {sorted(vs)}")
            print("  run: python scripts/stamp_version.py")
            sys.exit(1)
        print(f"OK: every asset stamped v={vs.pop() if vs else '?'}")
        return
    v = int(args[0]) if args else current() + 1
    stamp(v)
    print(f"stamped v={v} across index.html and every module import")


if __name__ == "__main__":
    main()
