import os
import time
import threading
from urllib.parse import quote
from dataclasses import dataclass

import cv2
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CameraConfig:
    name: str
    ip: str
    username: str
    password: str
    port: str
    stream_path: str

    @property
    def rtsp_url(self) -> str:
        user = quote(self.username, safe="")
        pwd = quote(self.password, safe="")
        return f"rtsp://{user}:{pwd}@{self.ip}:{self.port}/{self.stream_path}"


def load_camera_configs() -> list[CameraConfig]:
    username = os.environ["CAMERA_USERNAME"]
    password = os.environ["CAMERA_PASSWORD"]
    port = os.environ["CAMERA_PORT"]
    stream_path = os.environ["CAMERA_STREAM_PATH"]

    configs = []
    index = 1
    while True:
        ip = os.environ.get(f"CAMERA{index}_IP")
        if not ip:
            break
        configs.append(
            CameraConfig(
                name=f"CAMERA{index}",
                ip=ip.strip(),
                username=username,
                password=password,
                port=port,
                stream_path=stream_path,
            )
        )
        index += 1
    return configs


class CameraStream:
    """Reads frames from one RTSP feed in a background thread and always
    exposes the latest frame, reconnecting automatically if the feed drops."""

    def __init__(self, config: CameraConfig, reconnect_delay: float = 3.0):
        self.config = config
        self.reconnect_delay = reconnect_delay
        self._capture = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._capture:
            self._capture.release()

    def _connect(self):
        capture = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self):
        while self._running:
            self._capture = self._connect()
            if not self._capture.isOpened():
                print(f"[{self.config.name}] failed to open {self.config.ip}, retrying in {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)
                continue

            print(f"[{self.config.name}] connected to {self.config.ip}")
            while self._running:
                ok, frame = self._capture.read()
                if not ok:
                    print(f"[{self.config.name}] lost connection, reconnecting")
                    break
                with self._lock:
                    self._frame = frame

            self._capture.release()
            if self._running:
                time.sleep(self.reconnect_delay)

    def get_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class CameraManager:
    def __init__(self):
        self.streams: dict[str, CameraStream] = {}

    def start_all(self):
        for config in load_camera_configs():
            self.streams[config.name] = CameraStream(config).start()
        return self

    def stop_all(self):
        for stream in self.streams.values():
            stream.stop()

    def get_frame(self, name: str):
        stream = self.streams.get(name)
        return stream.get_frame() if stream else None

    def get_frames(self) -> dict:
        return {name: stream.get_frame() for name, stream in self.streams.items()}


if __name__ == "__main__":
    manager = CameraManager().start_all()

    if not manager.streams:
        print("No CAMERA*_IP entries found in .env")
        raise SystemExit(1)

    try:
        while True:
            for name, frame in manager.get_frames().items():
                if frame is not None:
                    cv2.imshow(name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        manager.stop_all()
        cv2.destroyAllWindows()