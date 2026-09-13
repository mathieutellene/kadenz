"""FastAPI backend: serves the dashboard, handles uploads and the WebSocket.

Client -> server:
  {"type": "start", "file": "clip.mp4"} | {"type": "start", "webcam": true}
  {"type": "overlay", "heat": bool} | {"type": "stop"}
Server -> client: AnalysisSession messages as-is (hello/metrics/dj/tracks/...),
plus video frames as binary (4-byte timestamp + JPEG).

A single session and a single pump live in `state`: the LAST WebSocket to
connect adopts the running session, so a browser reload never orphans it.
"""
import asyncio
import json
import queue
import shutil
from pathlib import Path

import yaml
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .session import AnalysisSession

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "data" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

app = FastAPI(title="Kadenz")
app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")

# Una unica sesion y un unico pump: la ULTIMA conexion WS adopta la sesion en
# curso (si el navegador se reconecta —F5, hipo de red— no se queda huerfano).
state = {"session": None, "queue": None, "pump": None}


def load_cfg():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


def wipe_uploads():
    """Los videos analizados NO se conservan: solo vive el ultimo subido.

    Se limpia al arrancar el servidor y antes de cada subida nueva. Si un archivo
    sigue abierto por una sesion en marcha, se ignora y caera en la siguiente pasada.
    """
    if not UPLOADS.exists():
        return
    for item in UPLOADS.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except OSError:
            pass


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename).name
    if Path(name).suffix.lower() not in VIDEO_EXT:
        return {"ok": False, "error": "Formato no soportado"}
    if state["session"] is not None:
        state["session"].stop()  # libera el video anterior antes de borrarlo
    wipe_uploads()
    dest = UPLOADS / name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"ok": True, "name": name}


def _make_pump(ws, q, sess):
    async def pump():
        loop = asyncio.get_event_loop()
        send_dt = 1.0 / load_cfg()["video"]["send_fps"]

        async def send_frames():
            # Frames como BINARIO (4 bytes t_ms big-endian + JPEG), siempre el ultimo:
            # si el navegador va lento se saltan frames, nunca se acumula retraso.
            last_ms = -1
            while True:
                lf = sess.latest_frame
                if lf is not None and lf[0] != last_ms:
                    last_ms = lf[0]
                    await ws.send_bytes(last_ms.to_bytes(4, "big") + lf[1])
                await asyncio.sleep(send_dt)

        async def send_msgs():
            while True:
                try:
                    msg = await loop.run_in_executor(None, q.get, True, 0.4)
                except queue.Empty:
                    continue
                await ws.send_text(json.dumps(msg))

        await asyncio.gather(send_frames(), send_msgs())
    return pump


def _attach(ws):
    """Este WS adopta la sesion en curso: cancela el pump anterior y crea el suyo."""
    if state["pump"] is not None:
        state["pump"].cancel()
        state["pump"] = None
    if state["session"] is not None and state["session"].is_alive() and state["queue"] is not None:
        state["pump"] = asyncio.create_task(_make_pump(ws, state["queue"], state["session"])())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _attach(ws)  # si hay sesion en marcha, esta conexion la adopta (reconexiones/F5)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                cmd = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = cmd.get("type")
            if kind == "start":
                if state["session"] is not None:
                    state["session"].stop()
                cfg = load_cfg()
                if cmd.get("webcam"):
                    spec = int(cfg["video"].get("webcam_index", 0))
                    name = "Camara en vivo"
                else:
                    path = UPLOADS / Path(cmd.get("file", "")).name
                    if not path.exists():
                        await ws.send_text(json.dumps(
                            {"type": "error", "text": f"No existe el video: {path.name}"}
                        ))
                        continue
                    spec = str(path)
                    name = path.name
                q = queue.Queue(maxsize=120)
                sess = AnalysisSession(spec, cfg, q, name=name)
                sess.heat = bool(cmd.get("heat"))
                state["session"] = sess
                state["queue"] = q
                sess.start()
                _attach(ws)
            elif kind == "overlay":
                if state["session"] is not None:
                    state["session"].heat = bool(cmd.get("heat"))
            elif kind == "stop":
                if state["session"] is not None:
                    state["session"].stop()
    except WebSocketDisconnect:
        pass
    # OJO: al desconectarse un WS NO se para la sesion ni el pump global —
    # otra conexion (o una futura) puede seguir adoptandola.
