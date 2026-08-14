import os
import sys
import time
import math
import json
import argparse
import threading
from urllib.parse import quote
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv
from alert_emailer import ViolationEmailer

# Force OpenCV FFmpeg backend to use RTSP over TCP for reliable local LAN camera access
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Load environment credentials from parent folder if present
load_dotenv(dotenv_path=os.path.join("..", ".env"))
load_dotenv()

# Global Configuration & Paths
YOLO_MODEL = None
DETECT_ENABLED = True
EDIT_ROI_MODE = False
EDIT_LINE_MODE = False
MOUSE_DRAGGING = False
MOUSE_START_PT = None
MOUSE_CURRENT_PT = None

ROI_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "scanner_roi.json")
LOG_JSONL_FILE = os.path.join(os.path.dirname(__file__), "attendance_events.jsonl")
LOG_TXT_FILE = os.path.join(os.path.dirname(__file__), "attendance_events.log")
ALERTS_DIR = os.path.join(os.path.dirname(__file__), "unpunched_alerts")
os.makedirs(ALERTS_DIR, exist_ok=True)


def load_scanner_roi_config():
    default_cfg = {
        "scanner_roi": [1400, 550, 1920, 850],
        "entry_line_y": 640,  # Entry threshold line
        "entry_direction": "down",
        "fingerprint_timeout_sec": 3.0,
        "touch_required_sec": 0.1,
        "violation_grace_sec": 3.5,
    }
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, "r") as f:
                data = json.load(f)
                default_cfg.update(data)
        except Exception as e:
            print(f"[WARNING] Could not read {ROI_CONFIG_FILE}: {e}")
    return default_cfg


def save_scanner_roi_config(cfg):
    try:
        with open(ROI_CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[SUCCESS] Saved configuration to {ROI_CONFIG_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save {ROI_CONFIG_FILE}: {e}")


def log_attendance_event(event_data):
    try:
        with open(LOG_JSONL_FILE, "a") as f:
            f.write(json.dumps(event_data) + "\n")
        with open(LOG_TXT_FILE, "a") as f:
            f.write(f"[{event_data['timestamp']}] ID:{event_data['track_id']} | DIR:{event_data['direction']} | RESULT:{event_data['result']} | FINGERPRINT:{event_data['fingerprint_used']}\n")
        print(f"[EVENT LOGGED] ID:{event_data['track_id']} | Direction:{event_data['direction']} | Result:{event_data['result']}")
    except Exception as e:
        print(f"[WARNING] Failed to log event: {e}")


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


def extract_hand_kinematics(keypoints, keypoint_confs, poly_np=None, poly_min_x=None, poly_min_y=None, poly_max_y=None, min_conf=0.25):
    """
    Computes an 8-point articulated hand/arm kinematic model for both left and right arms:
    1. Elbow joint
    2. Wrist base
    3. Palm center
    4. Knuckle joint
    5. Index fingertip
    6. Middle fingertip
    7. Thumb joint
    8. Thumb tip
    Classifies hand intent: TOUCHING (on scanner), REACHING (moving towards scanner), or DOWN (idle).
    """
    hands = []
    if keypoints is None or len(keypoints) < 11:
        return hands

    arms = [(10, 8, "Right"), (9, 7, "Left")]

    for wrist_idx, elbow_idx, side in arms:
        w_conf = float(keypoint_confs[wrist_idx]) if keypoint_confs is not None else 1.0
        if w_conf < min_conf:
            continue

        wx, wy = float(keypoints[wrist_idx][0]), float(keypoints[wrist_idx][1])
        e_conf = float(keypoint_confs[elbow_idx]) if keypoint_confs is not None else 0.0

        if e_conf >= min_conf:
            ex, ey = float(keypoints[elbow_idx][0]), float(keypoints[elbow_idx][1])
            vx, vy = wx - ex, wy - ey
            L = math.hypot(vx, vy)
            if L > 10:
                ux, uy = vx / L, vy / L
                nx, ny = (-uy, ux) if side == "Right" else (uy, -ux)
            else:
                ux, uy, nx, ny = 1.0, 0.0, 0.0, 1.0
                L = 60.0
        else:
            ux, uy, nx, ny = 1.0, 0.0, 0.0, 1.0
            L = 60.0
            ex, ey = wx - 60.0, wy

        # 8-Point Hand Constellation:
        pt_wrist = (wx, wy)
        pt_palm = (wx + ux * 0.18 * L, wy + uy * 0.18 * L)
        pt_knuckle = (wx + ux * 0.32 * L, wy + uy * 0.32 * L)
        pt_index_tip = (wx + ux * 0.46 * L, wy + uy * 0.46 * L)
        pt_middle_tip = (wx + ux * 0.50 * L, wy + uy * 0.50 * L)
        pt_thumb_joint = (wx + ux * 0.15 * L + nx * 0.16 * L, wy + uy * 0.15 * L + ny * 0.16 * L)
        pt_thumb_tip = (wx + ux * 0.32 * L + nx * 0.24 * L, wy + uy * 0.32 * L + ny * 0.24 * L)
        pt_pinky = (wx + ux * 0.22 * L - nx * 0.14 * L, wy + uy * 0.22 * L - ny * 0.14 * L)

        all_points = [
            pt_wrist, pt_palm, pt_knuckle, pt_index_tip,
            pt_middle_tip, pt_thumb_joint, pt_thumb_tip, pt_pinky
        ]

        # Classify Hand Intent:
        is_touching = False
        is_reaching = False
        if poly_np is not None and poly_min_x is not None:
            is_touching = any(
                (cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), True) >= 0.0 and pt[0] >= (poly_min_x - 10))
                for pt in all_points
            )
            if poly_min_y is not None and poly_max_y is not None:
                is_reaching = (wx >= (poly_min_x - 130)) and (poly_min_y - 60 <= wy <= poly_max_y + 70)

        if is_touching:
            hand_state = "PUNCHING ✓"
        elif is_reaching:
            hand_state = "REACHING"
        else:
            hand_state = "DOWN"

        hands.append({
            "side": side,
            "elbow": (ex, ey),
            "wrist": pt_wrist,
            "palm": pt_palm,
            "knuckle": pt_knuckle,
            "thumb_joint": pt_thumb_joint,
            "thumb_tip": pt_thumb_tip,
            "index_tip": pt_index_tip,
            "middle_tip": pt_middle_tip,
            "pinky": pt_pinky,
            "fingertips": [pt_thumb_tip, pt_index_tip, pt_middle_tip],
            "all_points": all_points,
            "unit_vec": (ux, uy),
            "conf": w_conf,
            "length": L,
            "is_touching": is_touching,
            "is_reaching": is_reaching,
            "hand_state": hand_state,
        })

    return hands


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


def load_camera_configs(selected_cam=1) -> CameraConfig:
    """Loads door camera configuration from .env file."""
    username = os.environ.get("CAMERA_USERNAME", "admin")
    password = os.environ.get("CAMERA_PASSWORD", "Gwc@2026")
    port = os.environ.get("CAMERA_PORT", "554")
    stream_path = os.environ.get("CAMERA_STREAM_PATH", "Streaming/Channels/101")

    ip = os.environ.get(f"CAMERA{selected_cam}_IP") or os.environ.get("CAMERA1_IP", "192.168.90.20")
    return CameraConfig(
        name=f"CAMERA{selected_cam}",
        ip=ip.strip(),
        username=username,
        password=password,
        port=port,
        stream_path=stream_path,
    )


class DirectionAwareAttendanceTracker:
    """
    State Machine Tracker for Front Door Fingerprint Compliance:
    - Multi-Point Hand & Finger Motion Tracking (8 articulated points per hand)
    - Per-Hand Punch Intent Tracking (PUNCHING vs REACHING vs DOWN)
    - Instant Violation Trigger when person crosses with hands down (no punch attempt)
    - Duplicate Prevention: Each track_id is processed and logged EXACTLY ONCE.
    """

    def __init__(self, config=None):
        self.cfg = config or load_scanner_roi_config()
        self.scanner_roi = self.cfg["scanner_roi"]
        self.entry_line_y = self.cfg["entry_line_y"]
        self.entry_direction = self.cfg["entry_direction"]  # "down" or "up"
        self.timeout_sec = self.cfg.get("fingerprint_timeout_sec", 3.0)
        self.touch_required_sec = self.cfg.get("touch_required_sec", 0.1)
        self.violation_grace_sec = self.cfg.get("violation_grace_sec", 1.5)

        self.scanner_polygon = self.cfg.get("scanner_polygon", [
            [int(1920 * 0.885), int(1080 * 0.635)],
            [int(1920 * 0.962), int(1080 * 0.585)],
            [int(1920 * 0.948), int(1080 * 0.720)]
        ])
        self.tracked_persons = {}  # track_id -> dict state
        self.next_track_id = 1
        self.verified_punches_count = 0
        self.missed_punches_count = 0
        self.lock = threading.Lock()

    def update_config(self, key, value):
        with self.lock:
            self.cfg[key] = value
            if key == "scanner_roi":
                self.scanner_roi = value
            elif key == "scanner_polygon":
                self.scanner_polygon = value
            elif key == "entry_line_y":
                self.entry_line_y = value
            elif key == "violation_grace_sec":
                self.violation_grace_sec = value
            save_scanner_roi_config(self.cfg)

    def get_approach_zone(self):
        s_x1, s_y1, s_x2, s_y2 = self.scanner_roi
        return [max(0, s_x1 - 250), max(0, s_y1 - 150), s_x2 + 250, s_y2 + 350]

    def update(self, detected_people):
        now = time.time()
        with self.lock:
            updated_results = []
            line_y = self.entry_line_y
            poly_np = np.array(self.scanner_polygon, np.int32)
            poly_min_x = min(pt[0] for pt in self.scanner_polygon)
            poly_max_x = max(pt[0] for pt in self.scanner_polygon)
            poly_min_y = min(pt[1] for pt in self.scanner_polygon)
            poly_max_y = max(pt[1] for pt in self.scanner_polygon)

            for p in detected_people:
                px1, py1, px2, py2 = p["box"]
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                raw_track_id = p.get("track_id")

                # --- 1. MULTI-POINT HAND KINEMATICS & PER-HAND INTENT ---
                hand_models = extract_hand_kinematics(
                    p.get("keypoints"), p.get("keypoint_confs"),
                    poly_np, poly_min_x, poly_min_y, poly_max_y, min_conf=0.25
                )

                is_touching_scanner = any(h.get("is_touching", False) for h in hand_models)
                is_reaching_scanner = any(h.get("is_reaching", False) for h in hand_models)

                # Format hand status summaries (e.g. Left: DOWN | Right: REACHING)
                l_hand_status = "DOWN"
                r_hand_status = "DOWN"
                for h in hand_models:
                    if h["side"] == "Left":
                        l_hand_status = h["hand_state"]
                    elif h["side"] == "Right":
                        r_hand_status = h["hand_state"]

                # Matching ID: use YOLO tracker ID if available, otherwise match by Centroid + IoU
                matched_id = raw_track_id
                if matched_id is None:
                    best_cost = 9999.0
                    for tid, state in self.tracked_persons.items():
                        ecx, ecy = state["last_centroid"]
                        past_box = state["last_box"]
                        iou = compute_iou([px1, py1, px2, py2], past_box)
                        c_dist = math.hypot(pcx - ecx, pcy - ecy)
                        cost = (1.0 - iou) * 120 + c_dist
                        if cost < 200 and cost < best_cost:
                            best_cost = cost
                            matched_id = tid

                if matched_id is None:
                    matched_id = self.next_track_id
                    self.next_track_id += 1

                if matched_id not in self.tracked_persons:
                    init_dir = "LEAVING" if (pcy >= (line_y - 20) and not is_touching_scanner) else "ENTERING"
                    init_status = "IGNORED" if init_dir == "LEAVING" else "APPROACHING"

                    self.tracked_persons[matched_id] = {
                        "first_seen": now,
                        "last_seen": now,
                        "first_centroid": (pcx, pcy),
                        "positions": deque([(pcx, pcy)], maxlen=20),
                        "hand_trail": deque(maxlen=15),
                        "direction": init_dir,
                        "status": init_status,
                        "has_punched": False,
                        "touch_start": now if is_touching_scanner else None,
                        "touch_last_seen": now if is_touching_scanner else None,
                        "touch_duration": 0.0,
                        "event_logged": False,
                        "snapshot_saved": False,
                        "last_centroid": (pcx, pcy),
                        "last_box": [px1, py1, px2, py2],
                        "conf": p["conf"],
                        "line_crossed_at": None,
                        "hand_speed": 0.0,
                        "l_hand": l_hand_status,
                        "r_hand": r_hand_status,
                    }

                state = self.tracked_persons[matched_id]
                state["last_seen"] = now
                state["last_centroid"] = (pcx, pcy)
                state["last_box"] = [px1, py1, px2, py2]
                state["positions"].append((pcx, pcy))
                state["conf"] = p["conf"]
                state["l_hand"] = l_hand_status
                state["r_hand"] = r_hand_status

                # Track Hand Movement Velocity & Trail
                if hand_models:
                    lead_wrist = hand_models[0]["wrist"]
                    if state["hand_trail"]:
                        last_hx, last_hy, last_ht = state["hand_trail"][-1]
                        dt = max(now - last_ht, 1e-4)
                        move_dist = math.hypot(lead_wrist[0] - last_hx, lead_wrist[1] - last_hy)
                        state["hand_speed"] = round(move_dist / dt, 1)
                    state["hand_trail"].append((lead_wrist[0], lead_wrist[1], now))

                first_cx, first_cy = state["first_centroid"]
                dy_total = pcy - first_cy  # positive = moving DOWN into office, negative = moving UP away

                # --- 2. STRICT DIRECTION DETERMINATION ---
                if is_touching_scanner:
                    state["direction"] = "ENTERING"
                elif first_cy >= (line_y - 20) or dy_total < -20:
                    state["direction"] = "LEAVING"
                    state["status"] = "IGNORED"
                    state["line_crossed_at"] = None
                elif dy_total >= 0 and first_cy < line_y:
                    state["direction"] = "ENTERING"

                # --- 3. COMPLIANCE & TOUCH PROCESSING ---
                # Require sustained presence of hand on the scanner pad (>= 0.25s)
                if is_touching_scanner:
                    if state["touch_start"] is None:
                        state["touch_start"] = now
                    state["touch_last_seen"] = now
                    state["touch_duration"] = now - state["touch_start"]

                    if state["touch_duration"] >= 0.25 and not state["has_punched"]:
                        state["has_punched"] = True
                        state["status"] = "COMPLIANT"
                        self.verified_punches_count += 1

                        if not state["event_logged"]:
                            state["event_logged"] = True
                            log_attendance_event({
                                "track_id": matched_id,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "direction": "ENTERING",
                                "fingerprint_used": True,
                                "result": "COMPLIANT",
                                "conf": round(p["conf"], 2),
                                "snapshot": None
                            })
                else:
                    if state["touch_start"] is not None and not state["has_punched"]:
                        time_away = now - (state.get("touch_last_seen") or state["touch_start"])
                        if time_away > 0.4:
                            state["touch_start"] = None
                            state["touch_duration"] = 0.0

                # --- 4. PER-HAND VIOLATION DETECTION (ENTERING ONLY) ---
                if state["direction"] == "ENTERING" and first_cy < line_y:
                    if state["has_punched"]:
                        state["status"] = "COMPLIANT"
                    else:
                        crossed_line = (pcy >= line_y) or (py2 >= (line_y + 60))
                        if crossed_line:
                            if state["line_crossed_at"] is None:
                                state["line_crossed_at"] = now

                            grace_elapsed = now - state["line_crossed_at"]

                            # SMART HAND-INTENT EVALUATION:
                            # 1) If hands are DOWN (idle at waist, swinging, no reaching) and person is crossing into the room:
                            #    -> Trigger VIOLATION immediately (e.g. at pcy >= line_y + 30 or grace >= 0.5s)
                            # 2) If a hand IS reaching towards scanner:
                            #    -> Allow brief 1.2s grace to touch the pad before concluding violation
                            hands_down = (not is_reaching_scanner) and (not is_touching_scanner)
                            hands_down_entering = hands_down and ((pcy >= line_y + 30) or (grace_elapsed >= 0.5))
                            walked_past = (grace_elapsed >= 1.2) or (pcy >= line_y + 120)

                            if hands_down_entering or walked_past:
                                state["status"] = "VIOLATION"
                                if not state["event_logged"]:
                                    state["event_logged"] = True
                                    self.missed_punches_count += 1
                                    state["should_snapshot"] = True
                                    state["snapshot_saved"] = True
                                    snapshot_name = f"UNPUNCHED_ENTRY_Track{matched_id}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                                    snapshot_path = os.path.join(ALERTS_DIR, snapshot_name)
                                    log_attendance_event({
                                        "track_id": matched_id,
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "direction": "ENTERING",
                                        "fingerprint_used": False,
                                        "result": "VIOLATION (hands down / unpunched)",
                                        "conf": round(p["conf"], 2),
                                        "snapshot": snapshot_path
                                    })
                            else:
                                state["status"] = "CHECKING"
                        else:
                            state["status"] = "APPROACHING"
                elif state["direction"] == "LEAVING":
                    state["status"] = "IGNORED"

                should_snap = state.get("should_snapshot", False)
                state["should_snapshot"] = False

                updated_results.append({
                    "track_id": matched_id,
                    "box": [px1, py1, px2, py2],
                    "conf": p["conf"],
                    "direction": state["direction"],
                    "status": state["status"],
                    "touch_sec": round(state["touch_duration"], 1),
                    "should_snapshot": should_snap,
                    "l_hand": state["l_hand"],
                    "r_hand": state["r_hand"],
                })

            # --- 5. TRACK CLEANUP ---
            expired_ids = [tid for tid, s in self.tracked_persons.items() if now - s["last_seen"] > 4.0]
            for tid in expired_ids:
                s = self.tracked_persons[tid]
                if (
                    s["direction"] == "ENTERING"
                    and s["first_centroid"][1] < line_y
                    and s["line_crossed_at"] is not None
                    and not s["has_punched"]
                    and not s["event_logged"]
                ):
                    s["event_logged"] = True
                    self.missed_punches_count += 1
                    snapshot_name = f"UNPUNCHED_ENTRY_Track{tid}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    snapshot_path = os.path.join(ALERTS_DIR, snapshot_name)
                    log_attendance_event({
                        "track_id": tid,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "direction": "ENTERING",
                        "fingerprint_used": False,
                        "result": "VIOLATION (walked inside)",
                        "conf": round(s["conf"], 2),
                        "snapshot": snapshot_path
                    })
                    print(f"[VIOLATION ALERT] Track {tid} entered room without biometric punch!")
                del self.tracked_persons[tid]

            return updated_results


attendance_tracker = DirectionAwareAttendanceTracker()

# Global emailer — initialised once, reused for every violation alert
_violation_emailer: ViolationEmailer | None = None


def get_violation_emailer(camera_name: str = "CAMERA1") -> ViolationEmailer:
    """Lazy-initialise the global emailer with the live camera name."""
    global _violation_emailer
    if _violation_emailer is None:
        _violation_emailer = ViolationEmailer(camera_name=camera_name)
    return _violation_emailer


def init_yolo_detector(model_name="yolo11n-pose.pt"):
    global YOLO_MODEL, DETECT_ENABLED
    try:
        from ultralytics import YOLO
        model_paths = [
            model_name,
            "yolo11n-pose.pt",
            "yolo11s-pose.pt",
            "yolov8n-pose.pt",
            os.path.join(".", model_name),
            os.path.join("..", "productiveity_analysis", model_name),
            "yolo26s.pt",
            "yolo11s.pt",
            "yolov8s.pt",
            "yolov8m.pt"
        ]
        chosen_path = None
        for p in model_paths:
            if os.path.exists(p):
                chosen_path = p
                break
        if not chosen_path:
            chosen_path = model_name

        print(f"[+] Loading Door Fingerprint AI Model: {chosen_path} ...")
        YOLO_MODEL = YOLO(chosen_path)
        DETECT_ENABLED = True
        print("[SUCCESS] Direction-Aware Fingerprint Compliance AI Engine Active!")
    except Exception as e:
        print(f"[WARNING] Could not load YOLO model: {e}")
        DETECT_ENABLED = False


def save_missed_person_snapshot(annotated_frame, track_id, camera_name="CAMERA1"):
    """Save violation snapshot and fire an email alert in the background."""
    try:
        os.makedirs(ALERTS_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        ts_human = time.strftime("%Y-%m-%d %H:%M:%S")
        filename = f"UNPUNCHED_ENTRY_Track{track_id}_{ts}.jpg"
        filepath = os.path.join(ALERTS_DIR, filename)
        ok = cv2.imwrite(filepath, annotated_frame)
        if ok:
            print(f"[ALERT SNAPSHOT SAVED] Captured unpunched entry alert -> {filepath}")
            get_violation_emailer(camera_name).send(
                snapshot_path=filepath,
                track_id=track_id,
                timestamp=ts_human
            )
        else:
            print(f"[ERROR] cv2.imwrite failed to write image file: {filepath}")
    except Exception as e:
        print(f"[WARNING] Could not save snapshot: {e}")


def on_mouse_event(event, x, y, flags, param):
    global MOUSE_DRAGGING, MOUSE_START_PT, MOUSE_CURRENT_PT, EDIT_ROI_MODE
    if not EDIT_ROI_MODE:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        MOUSE_DRAGGING = True
        MOUSE_START_PT = (x, y)
        MOUSE_CURRENT_PT = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if MOUSE_DRAGGING:
            MOUSE_CURRENT_PT = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if MOUSE_DRAGGING and MOUSE_START_PT is not None:
            MOUSE_DRAGGING = False
            x1 = min(MOUSE_START_PT[0], x)
            y1 = min(MOUSE_START_PT[1], y)
            x2 = max(MOUSE_START_PT[0], x)
            y2 = max(MOUSE_START_PT[1], y)

            if (x2 - x1) > 40 and (y2 - y1) > 40:
                attendance_tracker.update_config("scanner_roi", [x1, y1, x2, y2])
                print(f"[ROI UPDATED] New Scanner ROI set to: [{x1}, {y1}, {x2}, {y2}]")
            MOUSE_START_PT = None
            MOUSE_CURRENT_PT = None


def process_door_camera_frame(cam_name, frame):
    """
    Processes front entrance door camera stream:
    - Renders Virtual Entry Line & Biometric Scanner ROI.
    - Determines Person Direction: ENTERING vs LEAVING.
    - ENTERING -> Check Fingerprint Compliance -> COMPLIANT (Green) or VIOLATION (Red + Snapshot + Log).
    - LEAVING -> IGNORED (Slate Gray). No alerts, no compliance checks.
    """
    if not DETECT_ENABLED or YOLO_MODEL is None or frame is None:
        return frame

    try:
        annotated = frame.copy()
        h_img, w_img, _ = frame.shape

        # Recompute scanner triangle from actual frame dimensions (tightly wrapping the physical wall scanner)
        attendance_tracker.scanner_polygon = [
            [int(w_img * 0.885), int(h_img * 0.635)],  # Left edge of scanner mount
            [int(w_img * 0.962), int(h_img * 0.585)],  # Top-right — top of scanner on wall
            [int(w_img * 0.948), int(h_img * 0.720)]   # Bot-right — bottom of scanner on wall
        ]
        poly_xs = [pt[0] for pt in attendance_tracker.scanner_polygon]
        poly_ys = [pt[1] for pt in attendance_tracker.scanner_polygon]
        attendance_tracker.scanner_roi = [
            min(poly_xs), min(poly_ys), max(poly_xs), max(poly_ys)
        ]

        # Use YOLO tracking engine (ByteTrack if available) for persistent track IDs across frames
        try:
            res_all = YOLO_MODEL.track(frame, imgsz=640, conf=0.22, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        except Exception:
            res_all = YOLO_MODEL.predict(frame, imgsz=640, conf=0.22, verbose=False)[0]

        detected_people = []
        has_kpts = hasattr(res_all, "keypoints") and res_all.keypoints is not None

        if res_all.boxes:
            for idx, b in enumerate(res_all.boxes):
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
                w, h = x2 - x1, y2 - y1
                track_id = int(b.id[0].item()) if b.id is not None else None

                # Ignore non-person posters/art
                if 1400 <= x1 <= 1650 and y1 <= 200 and w < 110 and h > 150:
                    continue

                kpts_xy = None
                kpts_conf = None
                if has_kpts and len(res_all.keypoints.xy) > idx:
                    kpts_xy = res_all.keypoints.xy[idx].cpu().numpy()
                    if res_all.keypoints.conf is not None and len(res_all.keypoints.conf) > idx:
                        kpts_conf = res_all.keypoints.conf[idx].cpu().numpy()

                detected_people.append({
                    "conf": conf,
                    "box": [x1, y1, x2, y2],
                    "w": w,
                    "h": h,
                    "track_id": track_id,
                    "keypoints": kpts_xy,
                    "keypoint_confs": kpts_conf
                })

        # Run Direction-Aware Attendance Compliance Tracking
        results = attendance_tracker.update(detected_people)

        # 1. Draw Configurable Virtual Entry Line (Bright Yellow / Cyan)
        line_y = attendance_tracker.entry_line_y
        cv2.line(annotated, (0, line_y), (w_img, line_y), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"▼ VIRTUAL ENTRY LINE (ENTRY DIRECTION: DOWN) Y={line_y} ▼", (20, max(line_y - 8, 20)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw Biometric Scanner Zone — Cyan Triangle tightly wrapping the physical scanner
        tri = attendance_tracker.scanner_polygon
        poly_pts = np.array(tri, np.int32).reshape((-1, 1, 2))
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [poly_pts], (255, 200, 0))
        cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)
        cv2.polylines(annotated, [poly_pts], True, (255, 200, 0), 2, cv2.LINE_AA)
        left_pt = tri[0]
        label_x = max(left_pt[0] - 10, 10)
        label_y = max(left_pt[1] - 12, 20)
        cv2.putText(annotated, "BIOMETRIC FINGERPRINT", (label_x, label_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

        # Draw Multi-Point Articulated Hand Skeletons & Dynamic Motion Trails
        for p in detected_people:
            kpts = p.get("keypoints")
            kconfs = p.get("keypoint_confs")
            hand_models = extract_hand_kinematics(
                kpts, kconfs,
                poly_pts, min(poly_xs), min(poly_ys), max(poly_ys),
                min_conf=0.25
            )
            track_id = p.get("track_id")

            # 1. Render glowing motion trajectory trail
            if track_id is not None and track_id in attendance_tracker.tracked_persons:
                p_state = attendance_tracker.tracked_persons[track_id]
                trail = list(p_state.get("hand_trail", []))
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                    pt2 = (int(trail[i][0]), int(trail[i][1]))
                    trail_color = (int(74 * alpha), int(222 * alpha), int(255 * alpha))
                    cv2.line(annotated, pt1, pt2, trail_color, max(1, int(3 * alpha)), cv2.LINE_AA)

                # Display hand speed readout
                h_speed = p_state.get("hand_speed", 0.0)
                if trail and h_speed > 20:
                    lead_x, lead_y, _ = trail[-1]
                    cv2.putText(annotated, f"Hand: {h_speed} px/s", (int(lead_x + 12), int(lead_y - 8)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

            # 2. Render Articulated Hand Bones & 8-Point Constellation
            for hand in hand_models:
                ex, ey = int(hand["elbow"][0]), int(hand["elbow"][1])
                wx, wy = int(hand["wrist"][0]), int(hand["wrist"][1])
                px, py = int(hand["palm"][0]), int(hand["palm"][1])
                kx, ky = int(hand["knuckle"][0]), int(hand["knuckle"][1])
                itx, ity = int(hand["index_tip"][0]), int(hand["index_tip"][1])
                mtx, mty = int(hand["middle_tip"][0]), int(hand["middle_tip"][1])
                tjx, tjy = int(hand["thumb_joint"][0]), int(hand["thumb_joint"][1])
                ttx, tty = int(hand["thumb_tip"][0]), int(hand["thumb_tip"][1])
                pkx, pky = int(hand["pinky"][0]), int(hand["pinky"][1])

                # Check if any part of hand is touching scanner
                hand_touching = any(cv2.pointPolygonTest(poly_pts, (int(pt[0]), int(pt[1])), False) >= 0 for pt in hand["all_points"])
                bone_color = (74, 222, 128) if hand_touching else (255, 200, 0)
                bone_w = 2 if hand_touching else 1

                # Forearm bone
                cv2.line(annotated, (ex, ey), (wx, wy), bone_color, 2, cv2.LINE_AA)
                # Palm & Knuckle bones
                cv2.line(annotated, (wx, wy), (px, py), bone_color, bone_w, cv2.LINE_AA)
                cv2.line(annotated, (px, py), (kx, ky), bone_color, bone_w, cv2.LINE_AA)
                # Finger rays
                cv2.line(annotated, (kx, ky), (itx, ity), bone_color, bone_w, cv2.LINE_AA)
                cv2.line(annotated, (kx, ky), (mtx, mty), bone_color, bone_w, cv2.LINE_AA)
                cv2.line(annotated, (wx, wy), (tjx, tjy), bone_color, bone_w, cv2.LINE_AA)
                cv2.line(annotated, (tjx, tjy), (ttx, tty), bone_color, bone_w, cv2.LINE_AA)
                cv2.line(annotated, (wx, wy), (pkx, pky), bone_color, bone_w, cv2.LINE_AA)

                # Draw Hand Keypoint Nodes (8 points)
                for pt in hand["all_points"]:
                    p_x, p_y = int(pt[0]), int(pt[1])
                    is_inside = cv2.pointPolygonTest(poly_pts, (p_x, p_y), False) >= 0
                    node_color = (74, 222, 128) if is_inside else (0, 255, 255)
                    cv2.circle(annotated, (p_x, p_y), 4 if not is_inside else 6, node_color, -1, cv2.LINE_AA)
                    if is_inside:
                        cv2.circle(annotated, (p_x, p_y), 10, (74, 222, 128), 2, cv2.LINE_AA)

        # Draw mouse dragging rectangle when user is editing ROI
        if MOUSE_DRAGGING and MOUSE_START_PT and MOUSE_CURRENT_PT:
            mx1 = min(MOUSE_START_PT[0], MOUSE_CURRENT_PT[0])
            my1 = min(MOUSE_START_PT[1], MOUSE_CURRENT_PT[1])
            mx2 = max(MOUSE_START_PT[0], MOUSE_CURRENT_PT[0])
            my2 = max(MOUSE_START_PT[1], MOUSE_CURRENT_PT[1])
            cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (0, 255, 255), 2)
            cv2.putText(annotated, "DRAWING NEW SCANNER ROI...", (mx1, max(my1 - 8, 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Bounding Boxes with Track ID, Direction, and Compliance Status
        for r in results:
            x1, y1, x2, y2 = r["box"]
            status = r["status"]
            direction = r["direction"]
            touch_sec = r["touch_sec"]
            track_id = r["track_id"]
            should_snap = r.get("should_snapshot", False)

            l_h = r.get("l_hand", "DOWN")
            r_h = r.get("r_hand", "DOWN")

            if status == "COMPLIANT":
                color = (74, 222, 128)   # Lime Green
                label = f"ID:{track_id} | ENTERING | COMPLIANT ✓ ({touch_sec}s)"
                thickness = 2
            elif status == "VIOLATION":
                color = (0, 0, 255)      # Bold Red
                label = f"ID:{track_id} | ENTERING | VIOLATION! NO FINGERPRINT"
                thickness = 3
            elif status == "CHECKING":
                color = (0, 165, 255)    # Orange — crossed line, grace running
                label = f"ID:{track_id} | CHECKING (L:{l_h} | R:{r_h})"
                thickness = 2
            elif status == "IGNORED" or direction == "LEAVING":
                color = (148, 163, 184)  # Slate Gray
                label = f"ID:{track_id} | LEAVING (IGNORED)"
                thickness = 1
            else:  # APPROACHING
                color = (200, 200, 200)  # Light gray
                label = f"ID:{track_id} | APPROACHING (L:{l_h} | R:{r_h})"
                thickness = 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, color, 1, cv2.LINE_AA)

            # Highlight Red Banner if person violated attendance compliance
            if status == "VIOLATION":
                cv2.rectangle(annotated, (x1, y1), (x2, y1 + 30), (0, 0, 255), -1)
                cv2.putText(annotated, "ALERT: UNPUNCHED ENTRY!", (x1 + 10, y1 + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Auto-save snapshot + trigger email alert on violation
            if should_snap:
                save_missed_person_snapshot(annotated, track_id, cam_name)

        # 4. Render Top HUD Banner
        v_count = attendance_tracker.verified_punches_count
        m_count = attendance_tracker.missed_punches_count
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w_img, 50), (15, 23, 42), -1)
        cv2.addWeighted(hud_overlay, 0.75, annotated, 0.25, 0, annotated)

        if EDIT_ROI_MODE:
            hud_text = f"[ROI EDIT MODE] Click & Drag mouse to draw Scanner Box | Press 'E' to Exit"
            cv2.putText(annotated, hud_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        elif EDIT_LINE_MODE:
            hud_text = f"[ENTRY LINE EDIT MODE] Use UP / DOWN Arrow Keys to move Entry Line | Press 'L' to Exit"
            cv2.putText(annotated, hud_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            hud_text = f"FRONT DOOR MONITOR ({cam_name}) | COMPLIANT: {v_count} | VIOLATIONS: {m_count} | Keys: [E] ROI [L] Line [Q] Quit"
            cv2.putText(annotated, hud_text, (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated
    except Exception as e:
        import traceback
        traceback.print_exc()
        return frame


class AsyncCameraStream:
    """Reads frames from front door RTSP camera with zero latency and smooth rendering."""

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

            print(f"[ONLINE] Front Door Camera Stream Live! ({self.config.ip})")
            while self._running:
                try:
                    for _ in range(2):
                        self._capture.grab()
                    ok, frame = self._capture.retrieve()
                except Exception:
                    ok, frame = False, None

                if not ok or frame is None:
                    print(f"[{self.config.name}] Stream interrupted, reconnecting...")
                    break
                with self._lock:
                    self._raw_frame = frame

            self._capture.release()
            if self._running:
                time.sleep(self.reconnect_delay)

    def _run_ai(self):
        while self._running:
            frame_to_process = None
            with self._lock:
                if self._raw_frame is not None:
                    frame_to_process = self._raw_frame.copy()

            if frame_to_process is not None and DETECT_ENABLED:
                annotated = process_door_camera_frame(self.config.name, frame_to_process)
                with self._lock:
                    self._annotated_frame = annotated
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
    global EDIT_ROI_MODE, EDIT_LINE_MODE
    parser = argparse.ArgumentParser(description="Direction-Aware Front Door Fingerprint Attendance Compliance Detector")
    parser.add_argument("--cam", type=int, default=1, help="Door camera number from .env (default: 1)")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="YOLO model path")
    args = parser.parse_args()

    config = load_camera_configs(selected_cam=args.cam)
    # Pre-initialise emailer with the correct camera name
    get_violation_emailer(camera_name=config.name)

    print("=" * 75)
    print("  DIRECTION-AWARE FRONT DOOR FINGERPRINT ATTENDANCE COMPLIANCE AI DETECTOR")
    print("=" * 75)
    print(f" [+] Door Camera: {config.name} ({config.ip})")
    print(f" [+] Rule 1: ENTERING → Check Thumb/Fingerprint -> [GREEN] COMPLIANT or [RED] VIOLATION")
    print(f" [+] Rule 2: LEAVING  → [SLATE GRAY] IGNORED (Zero Alerts/Snapshots)")
    print(f" [+] Shortcuts: Press 'E' (Edit Scanner Box) | 'L' (Edit Virtual Entry Line) | 'Q' (Quit)")
    print("=" * 75 + "\n")

    # Initialize Model
    init_yolo_detector(model_name=args.model)

    # Start Async Camera Stream
    stream = AsyncCameraStream(config).start()

    print("[+] Launching Front Door Fingerprint Monitor... Press 'q' to EXIT.\n")

    window_name = f"FRONT DOOR FINGERPRINT MONITOR ({config.name})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    cv2.setMouseCallback(window_name, on_mouse_event)

    try:
        while True:
            frame = stream.get_display_frame()
            if frame is not None:
                cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[+] Exiting Direction-Aware Fingerprint Compliance Monitor...")
                break
            elif key == ord("e") or key == ord("E"):
                EDIT_ROI_MODE = not EDIT_ROI_MODE
                mode_str = "ENABLED" if EDIT_ROI_MODE else "DISABLED"
                print(f"[+] ROI Edit Mode {mode_str}. (Click & drag mouse on window to adjust scanner box)")
            elif key == ord("l") or key == ord("L"):
                EDIT_LINE_MODE = not EDIT_LINE_MODE
                mode_str = "ENABLED" if EDIT_LINE_MODE else "DISABLED"
                print(f"[+] Entry Line Edit Mode {mode_str}. (Use Up/Down arrow keys to adjust Virtual Entry Line)")
            elif EDIT_LINE_MODE and key == 82:  # Up Arrow
                attendance_tracker.update_config("entry_line_y", max(50, attendance_tracker.entry_line_y - 10))
                print(f"[ENTRY LINE UPDATED] Y = {attendance_tracker.entry_line_y}")
            elif EDIT_LINE_MODE and key == 84:  # Down Arrow
                attendance_tracker.update_config("entry_line_y", min(1080, attendance_tracker.entry_line_y + 10))
                print(f"[ENTRY LINE UPDATED] Y = {attendance_tracker.entry_line_y}")
            elif key == ord("r") or key == ord("R"):
                attendance_tracker.update_config("scanner_roi", [1400, 480, 1920, 900])
                attendance_tracker.update_config("entry_line_y", 620)
                print(f"[+] Reset Scanner ROI & Entry Line to default.")
    except KeyboardInterrupt:
        print("\n[+] Interrupted by user. Closing...")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("[+] Door camera stream stopped cleanly.")


if __name__ == "__main__":
    main()
