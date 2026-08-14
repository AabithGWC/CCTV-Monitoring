import os
import sys
import time
import subprocess
import threading
from urllib.parse import quote
from dataclasses import dataclass

import cv2
from dotenv import load_dotenv

# Force OpenCV FFmpeg backend to use RTSP over TCP for reliable local LAN camera access
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

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
    username = os.environ.get("CAMERA_USERNAME", "admin")
    password = os.environ.get("CAMERA_PASSWORD", "Gwc@2026")
    port = os.environ.get("CAMERA_PORT", "554")
    stream_path = os.environ.get("CAMERA_STREAM_PATH", "Streaming/Channels/101")

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


def ping_ip(ip: str) -> bool:
    """Quick 1-second ping check to verify Ethernet LAN reachability."""
    try:
        res = subprocess.call(
            ["ping", "-n", "1", "-w", "1000", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return res == 0
    except Exception:
        return False


def verify_lan_network(configs: list[CameraConfig]):
    """Check LAN reachability for all camera IPs before streaming."""
    print("=" * 65)
    print(" LAN ETHERNET CCTV CAMERA ACCESS CHECKER")
    print("=" * 65)
    print("[+] Checking LAN Ethernet reachability to camera IPs...")

    reachable_count = 0
    for cfg in configs:
        is_up = ping_ip(cfg.ip)
        status = "[ONLINE]" if is_up else "[OFFLINE / UNREACHABLE]"
        print(f"    - {cfg.name} [{cfg.ip}]: {status}")
        if is_up:
            reachable_count += 1

    print("-" * 65)
    if reachable_count == 0:
        print("[WARNING] NONE of the camera IPs are reachable over LAN Ethernet!")
        print("          Please ensure:")
        print("          1. Ethernet cable is securely plugged into your laptop.")
        print("          2. Laptop network adapter is set to 192.168.90.x subnet (or DHCP).")
        print("          3. Cameras are powered ON.")
        print("-" * 65)
    else:
        print(f"[SUCCESS] Connected to LAN! {reachable_count}/{len(configs)} cameras active.")
    print("=" * 65 + "\n")


class CameraStream:
    """Reads frames from one RTSP feed in a background thread and always
    exposes the latest frame, reconnecting automatically if the feed drops."""

    def __init__(self, config: CameraConfig, reconnect_delay: float = 2.0):
        self.config = config
        self.reconnect_delay = reconnect_delay
        self._capture = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.is_connected = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._capture:
            self._capture.release()

    def _connect(self):
        # Open stream with low buffer size for zero-latency live preview
        capture = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self):
        while self._running:
            self._capture = self._connect()
            if not self._capture.isOpened():
                self.is_connected = False
                print(f"[{self.config.name}] Connecting to {self.config.ip}... retrying in {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)
                continue

            self.is_connected = True
            print(f"[ONLINE] {self.config.name} Stream Live! ({self.config.ip})")
            while self._running:
                ok, frame = self._capture.read()
                if not ok:
                    print(f"[{self.config.name}] Signal lost / cable disconnected, reconnecting...")
                    self.is_connected = False
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
    def __init__(self, configs: list[CameraConfig]):
        self.configs = configs
        self.streams: dict[str, CameraStream] = {}

    def start_all(self):
        for config in self.configs:
            self.streams[config.name] = CameraStream(config).start()
        return self

    def stop_all(self):
        for stream in self.streams.values():
            stream.stop()

    def get_frames(self) -> dict:
        return {name: stream.get_frame() for name, stream in self.streams.items()}


def main():
    configs = load_camera_configs()
    if not configs:
        print("[ERROR] No CAMERA*_IP entries found in .env file!")
        sys.exit(1)

    # Perform initial LAN check
    verify_lan_network(configs)

    # Start multi-threaded camera streams
    manager = CameraManager(configs).start_all()

    print("[+] Launching Live Camera Windows... Press 'q' in any window to EXIT.")

    created_windows = set()

    try:
        while True:
            frames = manager.get_frames()
            for name, frame in frames.items():
                if frame is not None:
                    if name not in created_windows:
                        # WINDOW_NORMAL with 960x540 allows full zoomed-out frame display without cropping
                        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(name, 960, 540)
                        created_windows.add(name)
                    cv2.imshow(name, frame)

            # Wait 1ms; exit if 'q' key pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[+] Exiting LAN Camera Access...")
                break
    except KeyboardInterrupt:
        print("\n[+] Interrupted by user. Closing...")
    finally:
        manager.stop_all()
        cv2.destroyAllWindows()
        print("[+] All camera streams stopped cleanly.")


if __name__ == "__main__":
    main()
