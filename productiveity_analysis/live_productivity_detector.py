import os
import sys
import time
import math
import argparse
import threading
from urllib.parse import quote
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv

# Force OpenCV FFmpeg backend to use RTSP over TCP for reliable local LAN camera access
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Load environment credentials from parent folder if present
load_dotenv(dotenv_path=os.path.join("..", ".env"))
load_dotenv()

# Global AI & Idle Laptop Tracker Configuration
YOLO_MODEL = None
DETECT_ENABLED = True
IDLE_TIMEOUT_SECONDS = 1200  # 20 minutes default
USE_ZOOM_TILES = True  # Tiled Zoomed Sliced Inference for Small Object & Distant Person accuracy


def apply_nms_boxes(raw_detections, iou_thresh=0.45):
    """
    Applies class-aware Non-Maximum Suppression (NMS) to deduplicate overlapping bounding boxes.
    """
    if not raw_detections:
        return []

    final_detections = []
    classes = np.array([d["cls_id"] for d in raw_detections])
    unique_classes = set(classes)

    for c in unique_classes:
        class_indices = [i for i, d in enumerate(raw_detections) if d["cls_id"] == c]
        c_boxes = []
        c_scores = []
        for idx in class_indices:
            b = raw_detections[idx]["box"]
            c_boxes.append([b[0], b[1], b[2] - b[0], b[3] - b[1]])
            c_scores.append(raw_detections[idx]["conf"])

        indices = cv2.dnn.NMSBoxes(c_boxes, c_scores, score_threshold=0.01, nms_threshold=iou_thresh)
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append(raw_detections[class_indices[i]])

    return final_detections


def compute_iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = float(areaA + areaB - inter)
    return inter / max(union, 1e-6)


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


def load_camera_configs(selected_cams=None) -> list[CameraConfig]:
    """Loads camera configurations from .env, optionally filtering by camera numbers (e.g. [1, 2])."""
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
        if selected_cams is None or index in selected_cams:
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


class SmoothPersonTracker:
    """
    Butter-Smooth Person Box & State Tracking Engine:
    - Exponential Moving Average (EMA) box smoothing to eliminate bounding box jitter.
    - Classification Hysteresis (voting window) to prevent rapid red/green flickering.
    """

    def __init__(self, alpha=0.55):
        self.alpha = alpha  # EMA smoothing factor (0.55 = responsive yet silky smooth)
        self.tracks = {}  # cam_name -> {track_id -> dict}
        self.next_id = 1
        self.lock = threading.Lock()

    def update(self, cam_name, detected_people, laptops):
        now = time.time()
        with self.lock:
            if cam_name not in self.tracks:
                self.tracks[cam_name] = {}

            cam_tracks = self.tracks[cam_name]
            updated_results = []

            for p in detected_people:
                raw_box = p["box"]
                conf = p["conf"]
                px1, py1, px2, py2 = raw_box
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                w, h, ar = p["w"], p["h"], p["ar"]

                # 1. Match with existing track by IoU + Centroid distance
                matched_id = None
                best_cost = 9999.0

                for tid, state in cam_tracks.items():
                    ecx, ecy = state["last_centroid"]
                    past_box = state["smoothed_box"]
                    iou = compute_iou(raw_box, past_box)
                    c_dist = math.hypot(pcx - ecx, pcy - ecy)
                    cost = (1.0 - iou) * 100 + c_dist

                    if cost < 180 and cost < best_cost:
                        best_cost = cost
                        matched_id = tid

                # 2. Raw Classification Rule
                is_operating_laptop = False
                for l in laptops:
                    lx1, ly1, lx2, ly2 = l["box"]
                    lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                    dist = math.hypot(pcx - lcx, pcy - lcy)
                    if dist < 280 or (px1 - 40 <= lcx <= px2 + 40 and py1 <= lcy <= py2 + 80):
                        is_operating_laptop = True
                        break

                is_seated_body = (ar < 1.65)
                raw_is_roaming = not (is_operating_laptop or is_seated_body)

                if matched_id is None:
                    matched_id = self.next_id
                    self.next_id += 1
                    cam_tracks[matched_id] = {
                        "first_seen": now,
                        "last_seen": now,
                        "smoothed_box": [float(v) for v in raw_box],
                        "last_centroid": (pcx, pcy),
                        "status_history": deque([raw_is_roaming] * 5, maxlen=6),
                        "stable_status": "ROAMING" if raw_is_roaming else "WORKING",
                    }

                state = cam_tracks[matched_id]
                state["last_seen"] = now
                state["last_centroid"] = (pcx, pcy)
                state["status_history"].append(raw_is_roaming)

                # Exponential Moving Average Box Smoothing
                sb = state["smoothed_box"]
                new_sb = [
                    self.alpha * raw_box[i] + (1.0 - self.alpha) * sb[i]
                    for i in range(4)
                ]
                state["smoothed_box"] = new_sb

                # Hysteresis Voting for Classification Stability (Requires >= 4/6 frames agreement)
                roam_votes = sum(1 for v in state["status_history"] if v)
                if roam_votes >= 4:
                    state["stable_status"] = "ROAMING"
                elif roam_votes <= 2:
                    state["stable_status"] = "WORKING"

                int_smooth_box = [int(round(v)) for v in new_sb]

                updated_results.append({
                    "track_id": matched_id,
                    "box": int_smooth_box,
                    "conf": conf,
                    "status": state["stable_status"],
                })

            # Clean up old tracks (> 3s unseen)
            expired_ids = [tid for tid, s in cam_tracks.items() if now - s["last_seen"] > 3.0]
            for tid in expired_ids:
                del cam_tracks[tid]

            return updated_results


smooth_tracker = SmoothPersonTracker()


class IdleLaptopTracker:
    def __init__(self, timeout_sec=1200):
        self.timeout_sec = timeout_sec
        self.tracked_laptops = {}  # cam_name -> list of tracked laptops
        self.lock = threading.Lock()

    def update(self, cam_name, detected_laptops, detected_people):
        now = time.time()
        with self.lock:
            if cam_name not in self.tracked_laptops:
                self.tracked_laptops[cam_name] = []

            existing = self.tracked_laptops[cam_name]
            updated_list = []

            for l in detected_laptops:
                lx1, ly1, lx2, ly2 = l["box"]
                lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2

                is_occupied = False
                for p in detected_people:
                    px1, py1, px2, py2 = p["box"]
                    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                    dist = math.hypot(pcx - lcx, pcy - lcy)
                    if dist < 260 or (px1 - 40 <= lcx <= px2 + 40 and py1 <= lcy <= py2 + 80):
                        is_occupied = True
                        break

                matched_item = None
                for ext in existing:
                    elx1, ely1, elx2, ely2 = ext["bbox"]
                    elcx, elcy = (elx1 + elx2) / 2, (ely1 + ely2) / 2
                    if math.hypot(lcx - elcx, lcy - elcy) < 100:
                        matched_item = ext
                        break

                if matched_item:
                    if is_occupied:
                        matched_item["last_occupied"] = now
                        matched_item["unattended_sec"] = 0
                    else:
                        matched_item["unattended_sec"] = int(now - matched_item["last_occupied"])
                    matched_item["bbox"] = [lx1, ly1, lx2, ly2]
                    matched_item["conf"] = l["conf"]
                    matched_item["is_occupied"] = is_occupied
                    updated_list.append(matched_item)
                else:
                    new_item = {
                        "bbox": [lx1, ly1, lx2, ly2],
                        "conf": l["conf"],
                        "last_occupied": now,
                        "unattended_sec": 0,
                        "is_occupied": is_occupied,
                    }
                    updated_list.append(new_item)

            self.tracked_laptops[cam_name] = updated_list
            return updated_list


laptop_tracker = IdleLaptopTracker(timeout_sec=IDLE_TIMEOUT_SECONDS)


def init_yolo_detector(model_name="yolo26s.pt", idle_timeout_sec=1200, use_zoom=True):
    global YOLO_MODEL, DETECT_ENABLED, IDLE_TIMEOUT_SECONDS, USE_ZOOM_TILES, laptop_tracker
    IDLE_TIMEOUT_SECONDS = idle_timeout_sec
    USE_ZOOM_TILES = use_zoom
    laptop_tracker = IdleLaptopTracker(timeout_sec=idle_timeout_sec)
    try:
        from ultralytics import YOLO
        model_paths = [
            model_name,
            os.path.join(".", model_name),
            "yolo26n.pt",
            "yolo26s.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolo11n.pt"
        ]
        chosen_path = None
        for p in model_paths:
            if os.path.exists(p):
                chosen_path = p
                break
        if not chosen_path:
            chosen_path = model_name

        print(f"[+] Loading Local YOLO AI Model: {chosen_path} ...")
        YOLO_MODEL = YOLO(chosen_path)
        DETECT_ENABLED = True
        print(f"[SUCCESS] YOLO Employee Productivity AI Engine Active! (Idle Timeout: {idle_timeout_sec//60} mins | Zoom Mode: {'ON' if use_zoom else 'OFF'})")
    except Exception as e:
        print(f"[WARNING] Could not load YOLO model: {e}")
        DETECT_ENABLED = False


def run_ai_detection(cam_name, frame, fps=0.0):
    """
    Butter-Smooth Seated Workstation vs Roaming Person AI Classifier with Zoomed Tiled Inference:
    - Seated Desk Employees (at cubicles, desks, chairs, laptops): LIME GREEN (WORKING / SEATED)
    - Open Aisle / Corridor Walkers & Standing Roamers: BOLD RED (ROAMING)
    - Mobile Phones: CORAL (PHONE DISTRACTION)
    - 20-Min Idle Laptops: YELLOW/AMBER (IDLE LAPTOP > 20 MINS)
    """
    if not DETECT_ENABLED or YOLO_MODEL is None or frame is None:
        return frame

    try:
        annotated = frame.copy()
        h_img, w_img, _ = frame.shape
        
        raw_detections = []

        # 1. Full Frame Pass
        res_full = YOLO_MODEL.predict(frame, imgsz=640, conf=0.20, classes=[0, 63, 66, 67], verbose=False)[0]
        for b in res_full.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
            raw_detections.append({"cls_id": cls_id, "conf": conf, "box": [x1, y1, x2, y2]})

        # 2. Zoomed Sub-Tile Passes (if USE_ZOOM_TILES enabled)
        if USE_ZOOM_TILES:
            grid_y, grid_x = 2, 2
            tile_h, tile_w = int(h_img / grid_y), int(w_img / grid_x)
            overlap_y, overlap_x = int(tile_h * 0.20), int(tile_w * 0.20)

            for gy in range(grid_y):
                for gx in range(grid_x):
                    y_start = max(0, int(gy * tile_h - (overlap_y if gy > 0 else 0)))
                    x_start = max(0, int(gx * tile_w - (overlap_x if gx > 0 else 0)))
                    y_end = min(h_img, int((gy + 1) * tile_h + (overlap_y if gy < grid_y - 1 else 0)))
                    x_end = min(w_img, int((gx + 1) * tile_w + (overlap_x if gx < grid_x - 1 else 0)))

                    tile_crop = frame[y_start:y_end, x_start:x_end]
                    if tile_crop.shape[0] < 50 or tile_crop.shape[1] < 50:
                        continue

                    res_tile = YOLO_MODEL.predict(tile_crop, imgsz=640, conf=0.20, classes=[0, 63, 66, 67], verbose=False)[0]
                    for b in res_tile.boxes:
                        cls_id = int(b.cls[0].item())
                        conf = float(b.conf[0].item())
                        tx1, ty1, tx2, ty2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
                        raw_detections.append({
                            "cls_id": cls_id,
                            "conf": conf,
                            "box": [tx1 + x_start, ty1 + y_start, tx2 + x_start, ty2 + y_start]
                        })

        merged_detections = apply_nms_boxes(raw_detections, iou_thresh=0.45)

        laptops = []
        people = []
        phones = []

        for d in merged_detections:
            cls_id = d["cls_id"]
            conf = d["conf"]
            x1, y1, x2, y2 = d["box"]
            w, h = x2 - x1, y2 - y1

            # Ignore pillar wall art poster (Elon Musk / NOW poster)
            if 1400 <= x1 <= 1650 and y1 <= 200 and w < 110 and h > 150:
                continue

            if cls_id == 0:  # Person
                if conf >= 0.20 and h >= 40 and w >= 18 and (w * h >= 600):
                    people.append({
                        "conf": conf,
                        "box": [x1, y1, x2, y2],
                        "w": w,
                        "h": h,
                        "ar": round(h / max(w, 1), 2),
                    })
            elif cls_id in [63, 66] and h < 180 and w < 350:  # Laptop / Keyboard
                laptops.append({"box": [x1, y1, x2, y2], "conf": conf})
            elif cls_id == 67 and conf >= 0.20:  # Phone
                phones.append({"box": [x1, y1, x2, y2], "conf": conf})

        # Run Smooth Box & State Tracking
        tracked_people = smooth_tracker.update(cam_name, people, laptops)

        # Update 20-Min Idle Laptop Tracker
        tracked_laptops = laptop_tracker.update(cam_name, laptops, people)

        # 1. Draw Laptops (Cyan for Active, Yellow/Orange for IDLE > 20 Mins)
        idle_laptop_count = 0
        for tl in tracked_laptops:
            lx1, ly1, lx2, ly2 = tl["bbox"]
            unattended_sec = tl["unattended_sec"]
            is_idle_alert = unattended_sec >= IDLE_TIMEOUT_SECONDS

            if is_idle_alert:
                idle_laptop_count += 1
                color = (0, 215, 255)  # Amber/Yellow
                mins = unattended_sec // 60
                tag = f"IDLE LAPTOP ({mins}m > 20m)"
                cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, 3, cv2.LINE_AA)
            else:
                color = (238, 187, 36)  # Cyan
                tag = f"LAPTOP {int(tl['conf']*100)}%"
                cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, 2, cv2.LINE_AA)

            cv2.putText(annotated, tag, (lx1, max(ly1 - 5, 15)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1, cv2.LINE_AA)

        # 2. Draw Mobile Phones (Coral)
        phone_count = len(phones)
        for ph in phones:
            px1, py1, px2, py2 = ph["box"]
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (62, 114, 245), 3, cv2.LINE_AA)
            cv2.putText(annotated, f"PHONE DISTRACTION {int(ph['conf']*100)}%", (px1, max(py1 - 5, 15)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (62, 114, 245), 1, cv2.LINE_AA)

        # 3. Draw Smoothened People Bounding Boxes & Stable Classifications
        roaming_count = 0
        working_count = 0

        for p in tracked_people:
            x1, y1, x2, y2 = p["box"]
            status = p["status"]
            conf = p["conf"]

            if status == "ROAMING":
                roaming_count += 1
                color = (0, 0, 255)  # Bold Red
                label = "ROAMING"
                thickness = 3
            else:
                working_count += 1
                color = (74, 222, 128)  # Lime Green
                label = "PERSON"
                thickness = 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            cv2.putText(annotated, f"{label} {int(conf*100)}%", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1, cv2.LINE_AA)

        # 4. Render Live HUD Top Banner with Status Summary
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w_img, 50), (15, 23, 42), -1)
        cv2.addWeighted(hud_overlay, 0.75, annotated, 0.25, 0, annotated)

        hud_text = f"{cam_name} | FPS: {fps} | WORKING: {working_count} | ROAMING: {roaming_count} | PHONE USE: {phone_count} | IDLE LAPTOPS (>20m): {idle_laptop_count}"
        cv2.putText(annotated, hud_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
    except Exception as e:
        return frame


class AsyncCameraStream:
    """Reads frames from one RTSP camera with zero latency and smooth 30+ FPS rendering."""

    def __init__(self, config: CameraConfig, reconnect_delay: float = 2.0):
        self.config = config
        self.reconnect_delay = reconnect_delay
        self._capture = None
        self._raw_frame = None
        self._annotated_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._capture_thread = None
        self._ai_thread = None
        self.frame_counter = 0
        self.fps = 0.0
        self._fps_time = time.time()
        self._fps_frames = 0

    def start(self):
        self._running = True
        self._capture_thread = threading.Thread(target=self._run_capture, daemon=True)
        self._ai_thread = threading.Thread(target=self._run_ai, daemon=True)
        self._capture_thread.start()
        self._ai_thread.start()
        return self

    def stop(self):
        self._running = False
        if self._capture:
            self._capture.release()

    def _connect(self):
        capture = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run_capture(self):
        while self._running:
            self._capture = self._connect()
            if not self._capture.isOpened():
                print(f"[{self.config.name}] Connecting to {self.config.ip}... retrying in {self.reconnect_delay}s")
                time.sleep(self.reconnect_delay)
                continue

            print(f"[ONLINE] {self.config.name} Stream Live! ({self.config.ip})")
            while self._running:
                try:
                    for _ in range(3):
                        self._capture.grab()
                    ok, frame = self._capture.retrieve()
                except Exception:
                    ok, frame = False, None

                if not ok or frame is None:
                    print(f"[{self.config.name}] Signal lost or stream interrupted, reconnecting...")
                    break
                with self._lock:
                    self._raw_frame = frame
                    self.frame_counter += 1
                time.sleep(0.01)

            self._capture.release()
            if self._running:
                time.sleep(self.reconnect_delay)

    def _run_ai(self):
        last_processed = -1
        while self._running:
            frame_to_process = None
            current_counter = 0
            with self._lock:
                if self._raw_frame is not None and self.frame_counter != last_processed:
                    frame_to_process = self._raw_frame.copy()
                    current_counter = self.frame_counter

            if frame_to_process is not None and DETECT_ENABLED:
                self._fps_frames += 1
                now_t = time.time()
                dt = now_t - self._fps_time
                if dt >= 1.0:
                    self.fps = round(self._fps_frames / dt, 1)
                    self._fps_frames = 0
                    self._fps_time = now_t

                annotated = run_ai_detection(self.config.name, frame_to_process, fps=self.fps)
                with self._lock:
                    self._annotated_frame = annotated
                    last_processed = current_counter
                time.sleep(0.01)
            else:
                time.sleep(0.02)

    def get_display_frame(self):
        with self._lock:
            if self._annotated_frame is not None:
                return self._annotated_frame.copy()
            elif self._raw_frame is not None:
                return self._raw_frame.copy()
            return None


def main():
    parser = argparse.ArgumentParser(description="Live Multi-Camera CCTV Employee Productivity Detector")
    parser.add_argument("--cams", type=int, nargs="+", default=[1], help="Camera index list (e.g. --cams 1 2 4)")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="YOLO model path")
    parser.add_argument("--idle-timeout", type=int, default=1200, help="Idle laptop timeout in seconds (default: 1200s = 20 mins)")
    parser.add_argument("--enable-zoom", action="store_true", default=True, help="Enable zoomed tiled inference for small objects & distant people")
    parser.add_argument("--disable-zoom", action="store_false", dest="enable_zoom", help="Disable zoomed tiled inference")
    args = parser.parse_args()

    configs = load_camera_configs(selected_cams=args.cams)
    if not configs:
        print("[ERROR] No valid camera configuration found in .env!")
        return

    print("=" * 70)
    print("  LIVE REAL-TIME EMPLOYEE PRODUCTIVITY CCTV ANALYTICS ENGINE")
    print("=" * 70)
    print(f" [+] Selected Cameras: {[c.name for c in configs]}")
    print(f" [+] Idle Laptop Timeout: {args.idle_timeout // 60} mins ({args.idle_timeout}s)")
    print(f" [+] Zoomed Tiled Inference: {'ENABLED' if args.enable_zoom else 'DISABLED'}")
    print(f" [+] Feature 1: Working vs Roaming Person Classifier (Green/Red)")
    print(f" [+] Feature 2: Phone Usage Distraction Detector (Coral)")
    print(f" [+] Feature 3: 20-Min Idle Laptop Tracker (Yellow/Orange)")
    print(f" [+] Performance Engine: Fast Async Multi-Threaded 30+ FPS (Zero Lag)")
    print("=" * 70 + "\n")

    # Initialize YOLO Model
    init_yolo_detector(model_name=args.model, idle_timeout_sec=args.idle_timeout, use_zoom=args.enable_zoom)

    # Start Async Camera Streams
    streams = [AsyncCameraStream(cfg).start() for cfg in configs]

    print("[+] Launching Selected Live Camera Windows... Press 'q' in any window to EXIT.\n")

    # Create Display Windows
    for cfg in configs:
        cv2.namedWindow(cfg.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cfg.name, 960, 540)

    try:
        while True:
            for i, st in enumerate(streams):
                frame = st.get_display_frame()
                if frame is not None:
                    cv2.imshow(configs[i].name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[+] Exiting Live Employee Productivity Detector...")
                break
    except KeyboardInterrupt:
        print("\n[+] Interrupted by user. Closing...")
    finally:
        for st in streams:
            st.stop()
        cv2.destroyAllWindows()
        print("[+] All camera streams stopped cleanly.")


if __name__ == "__main__":
    main()
