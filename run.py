"""Arranca Kadenz: python run.py  ->  http://127.0.0.1:8765"""
import os
import sys
from pathlib import Path

import uvicorn
import yaml

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from engine.api import app, wipe_uploads  # noqa: E402

if __name__ == "__main__":
    wipe_uploads()  # los videos analizados no se conservan entre arranques
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    server = cfg["server"]
    print(f"Kadenz -> http://{server['host']}:{server['port']}")
    uvicorn.run(app, host=server["host"], port=server["port"], log_level="info")
