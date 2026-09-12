"""Video source abstraction.

A file path, a webcam index, an http:// phone stream or an rtsp:// IP camera are
interchangeable: nothing downstream knows where frames come from.
"""
import cv2


class VideoSource:
    def __init__(self, spec):
        self.spec = spec
        self.cap = None
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.is_stream = False

    def open(self):
        spec = self.spec
        if isinstance(spec, str) and spec.isdigit():
            spec = int(spec)
        self.is_stream = isinstance(spec, int) or str(spec).lower().startswith(("http", "rtsp"))
        if isinstance(spec, int):
            # Webcam en Windows: DirectShow abre mas rapido y falla menos que MSMF
            self.cap = cv2.VideoCapture(spec, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(spec)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            self.cap = cv2.VideoCapture(spec)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la fuente de video: {self.spec}"
                + (" (¿camara tapada, apagada o en uso por otra app?)"
                   if isinstance(spec, int) else "")
            )
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1.0 < fps <= 120.0 else 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return self

    @property
    def duration(self):
        if self.frame_count > 0 and not self.is_stream:
            return round(self.frame_count / self.fps, 2)
        return None

    def read(self):
        return self.cap.read()

    def grab(self):
        """Avanza un frame sin decodificarlo (barato) — para saltar cuando vamos atrasados."""
        return self.cap.grab()

    def release(self):
        if self.cap is not None:
            self.cap.release()
