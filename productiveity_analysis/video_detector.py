import os
import sys
import time
import json
import csv
import argparse
import glob
import math
import queue
import threading
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Class mapping for COCO dataset
COCO_CLASSES = {
    0: "person",
    63: "laptop",
    67: "cell phone"
}

# Color palette (BGR format for OpenCV)
CLASS_COLORS = {
    "working_person": (74, 222, 128),   # Lime Green for Seated / Active Workstation Person
    "roaming_person": (0, 0, 255),      # Bold Bright Red for Unwanted Roaming/Walking Person
    63: (238, 187, 36),                  # Cyan/Blue for Laptop
    67: (62, 114, 245),                  # Coral/Red-Orange for Cell Phone
    "default": (203, 166, 247)
}

def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=8):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    w, h = x2 - x1, y2 - y1
    r = min(r, w // 2, h // 2)
    if r < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return

    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_hud_overlay(frame, frame_idx, total_frames, fps, class_counts, alert_text=None):
    """Draw a translucent HUD header overlay matching frame resolution."""
    h, w, _ = frame.shape
    bar_height = 60
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    time_sec = frame_idx / max(fps, 1)
    mins = int(time_sec // 60)
    secs = int(time_sec % 60)
    time_str = f"TIME: {mins:02d}:{secs:02d}"
    frame_str = f"FRAME: {frame_idx}/{total_frames}"

    cv2.putText(frame, "CCTV MONITORING SYSTEM", (20, 25), 
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{time_str}  |  {frame_str}", (20, 48), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

    start_x = w - 580
    badges = [
        ("WORKING", class_counts.get("working_person", 0), (74, 222, 128)),
        ("ROAMING", class_counts.get("roaming_person", 0), (0, 0, 255)),
        ("LAPTOP", class_counts.get(63, 0), (238, 187, 36)),
        ("PHONE", class_counts.get(67, 0), (62, 114, 245))
    ]

    for label, count, color in badges:
        bg_color = (30, 41, 59) if count == 0 else (color[0]//3, color[1]//3, color[2]//3)
        text_color = (148, 163, 184) if count == 0 else color
        
        cv2.rectangle(frame, (start_x, 12), (start_x + 130, 48), bg_color, -1)
        cv2.rectangle(frame, (start_x, 12), (start_x + 130, 48), text_color, 1 if count == 0 else 2)
        
        badge_text = f"{label}: {count}"
        cv2.putText(frame, badge_text, (start_x + 10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        start_x += 142

    if alert_text:
        banner_overlay = frame.copy()
        cv2.rectangle(banner_overlay, (20, h - 55), (w - 20, h - 15), (15, 23, 42), -1)
        cv2.addWeighted(banner_overlay, 0.8, frame, 0.2, 0, frame)
        banner_color = (0, 0, 255) if "ROAMING" in alert_text else (62, 114, 245)
        cv2.rectangle(frame, (20, h - 55), (w - 20, h - 15), banner_color, 2)
        
        cv2.putText(frame, f"STATUS ALERT: {alert_text}", (35, h - 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

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

def detect_frame_objects(model, frame, conf_thresh=0.04, imgsz=1280, use_zoom=True, tile_grid=(2, 2), tile_overlap=0.20, nms_thresh=0.45):
    """
    High-Precision Object & Roaming Person Detector with Tiled Zoomed Sliced Inference:
    - Seated Cubicle & Desk Workers: LIME GREEN (PERSON)
    - Standing / Walking / Leaning Aisle Roamers: BOLD RED (ROAMING)
    - Mobile Phones: CORAL / RED-ORANGE
    - Printed Wall Posters / Pillar Art: 100% Filtered Out
    - Tiled Zoom Inference: Slices frame into grid patches to boost small object (phones, laptops, distant people) detection accuracy.
    """
    raw_detections = []

    # 1. Full Frame Pass (for global context)
    res_full = model.predict(frame, imgsz=imgsz, conf=conf_thresh, classes=[0, 63, 66, 67], verbose=False)[0]
    for b in res_full.boxes:
        cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
        raw_detections.append({"cls_id": cls_id, "conf": conf, "box": [x1, y1, x2, y2]})

    # 2. Zoomed Sub-Tile Passes (if use_zoom is enabled)
    if use_zoom:
        h, w, _ = frame.shape
        grid_y, grid_x = tile_grid
        tile_h = int(h / grid_y)
        tile_w = int(w / grid_x)
        overlap_y = int(tile_h * tile_overlap)
        overlap_x = int(tile_w * tile_overlap)

        for gy in range(grid_y):
            for gx in range(grid_x):
                y_start = max(0, int(gy * tile_h - (overlap_y if gy > 0 else 0)))
                x_start = max(0, int(gx * tile_w - (overlap_x if gx > 0 else 0)))
                y_end = min(h, int((gy + 1) * tile_h + (overlap_y if gy < grid_y - 1 else 0)))
                x_end = min(w, int((gx + 1) * tile_w + (overlap_x if gx < grid_x - 1 else 0)))

                tile_crop = frame[y_start:y_end, x_start:x_end]
                if tile_crop.shape[0] < 50 or tile_crop.shape[1] < 50:
                    continue

                res_tile = model.predict(tile_crop, imgsz=640, conf=conf_thresh, classes=[0, 63, 66, 67], verbose=False)[0]
                for b in res_tile.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    tx1, ty1, tx2, ty2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
                    # Map tile coordinates back to full frame space
                    raw_detections.append({
                        "cls_id": cls_id,
                        "conf": conf,
                        "box": [tx1 + x_start, ty1 + y_start, tx2 + x_start, ty2 + y_start]
                    })

    # 3. Deduplicate overlapping bounding boxes across tiles using Class-Aware NMS
    merged_detections = apply_nms_boxes(raw_detections, iou_thresh=nms_thresh)

    laptops = []
    people = []
    phones = []

    for d in merged_detections:
        cls_id = d["cls_id"]
        conf = d["conf"]
        x1, y1, x2, y2 = d["box"]
        w, h = x2 - x1, y2 - y1

        # Filter out Wall Poster on Right Pillar (Elon Musk / NOW poster)
        if 1400 <= x1 <= 1650 and y1 <= 200 and w < 110 and h > 150:
            continue

        if cls_id == 0:  # Person
            if conf >= 0.15:
                people.append({"conf": conf, "box": [x1, y1, x2, y2], "w": w, "h": h, "ar": round(h / max(w, 1), 2)})
        elif cls_id in [63, 66]:  # Laptop or Keyboard on desk
            if h < 180 and w < 350:
                laptops.append({"cls_id": 63, "conf": conf, "box": [x1, y1, x2, y2]})
        elif cls_id == 67:  # Cell phone
            if conf >= 0.15:
                phones.append({"cls_id": 67, "conf": conf, "box": [x1, y1, x2, y2]})

    current_boxes = []
    class_counts = {"working_person": 0, "roaming_person": 0, 0: 0, 63: len(laptops), 67: len(phones)}

    for l in laptops:
        current_boxes.append({
            "cls_id": 63,
            "class_name": "laptop",
            "confidence": round(l["conf"], 3),
            "bbox": l["box"],
            "is_roaming": False
        })

    for ph in phones:
        current_boxes.append({
            "cls_id": 67,
            "class_name": "cell phone",
            "confidence": round(ph["conf"], 3),
            "bbox": ph["box"],
            "is_roaming": False
        })

    for p in people:
        w, h, ar = p["w"], p["h"], p["ar"]
        x1, y1, x2, y2 = p["box"]
        pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2

        # 1. Laptop Operation Rule: Laptop MUST be directly in front of person's torso
        is_operating_laptop = False
        for l in laptops:
            lx1, ly1, lx2, ly2 = l["box"]
            lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            if (x1 - 25 <= lcx <= x2 + 25) and (y1 <= lcy <= y2 + 60):
                is_operating_laptop = True
                break

        # 2. Standing / Walking / Leaning Posture
        is_upright_posture = (ar >= 1.6 and h >= 120)

        if is_operating_laptop:
            is_roaming = False
        elif is_upright_posture:
            is_roaming = True
        else:
            # Compact seated cubicle workers
            if y2 < 450 and pcx < 1100 and h < 135:
                is_roaming = False
            else:
                is_roaming = True

        if is_roaming:
            class_counts["roaming_person"] += 1
        else:
            class_counts["working_person"] += 1
            
        class_counts[0] += 1

        current_boxes.append({
            "cls_id": 0,
            "class_name": "ROAMING" if is_roaming else "PERSON",
            "confidence": round(p["conf"], 3),
            "bbox": [x1, y1, x2, y2],
            "is_roaming": is_roaming
        })

    return current_boxes, class_counts

def smooth_update_boxes(current_boxes, target_boxes, factor=0.35):
    """
    Smoothly move current bounding boxes toward target bounding boxes to prevent jerky transitions.
    """
    if not target_boxes:
        return current_boxes
    if not current_boxes:
        return target_boxes

    result_boxes = []
    for tb in target_boxes:
        best_cb = None
        best_dist = 9999
        for cb in current_boxes:
            if cb["cls_id"] == tb["cls_id"]:
                dist = abs(cb["bbox"][0] - tb["bbox"][0]) + abs(cb["bbox"][1] - tb["bbox"][1])
                if dist < best_dist and dist < 120:
                    best_dist = dist
                    best_cb = cb

        if best_cb:
            cb_b = best_cb["bbox"]
            tb_b = tb["bbox"]
            smooth_b = [
                int(cb_b[0] + (tb_b[0] - cb_b[0]) * factor),
                int(cb_b[1] + (tb_b[1] - cb_b[1]) * factor),
                int(cb_b[2] + (tb_b[2] - cb_b[2]) * factor),
                int(cb_b[3] + (tb_b[3] - cb_b[3]) * factor)
            ]
            result_boxes.append({
                "cls_id": tb["cls_id"],
                "class_name": tb["class_name"],
                "confidence": tb["confidence"],
                "bbox": smooth_b,
                "is_roaming": tb.get("is_roaming", False)
            })
        else:
            result_boxes.append(tb)

    return result_boxes

def process_video(input_path, output_dir, model_path="yolov8m.pt", stride=2, conf_thresh=0.04, imgsz=1280, max_frames=None, start_frame=0, show=False, use_zoom=True):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[+] Loading YOLO model: {model_path} ...")
    model = YOLO(model_path)

    print(f"[+] Opening video file: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames -= start_frame

    if max_frames and max_frames < total_frames:
        total_frames = max_frames
        
    output_video_path = os.path.join(output_dir, "output_detection.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"[+] Resolution: {width}x{height} @ {fps:.2f} FPS | Total Frames to Process: {total_frames}")
    print(f"[+] Tiled Zoomed Sliced Inference Mode: {'ACTIVE' if use_zoom else 'DISABLED'}")
    print(f"[+] Saving annotated video to: {output_video_path}")

    if show:
        print("[+] Starting Async Threaded Live Player (Butter-Smooth 25+ FPS Rendering)...")
        frame_queue = queue.Queue(maxsize=2)
        result_queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()

        def inference_worker():
            while not stop_event.is_set():
                try:
                    f_data = frame_queue.get(timeout=0.05)
                    if f_data is None:
                        break
                    f_idx, frame_copy = f_data
                    boxes, counts = detect_frame_objects(model, frame_copy, conf_thresh=conf_thresh, imgsz=imgsz, use_zoom=use_zoom)
                    
                    while not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            break
                    result_queue.put((f_idx, boxes, counts))
                    frame_queue.task_done()
                except queue.Empty:
                    continue

        worker_thread = threading.Thread(target=inference_worker, daemon=True)
        worker_thread.start()

        cv2.namedWindow("CCTV Live Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CCTV Live Monitor", 1280, 720)

        frame_idx = 0
        latest_target_boxes = []
        displayed_boxes = []
        current_counts = {"working_person": 0, "roaming_person": 0, 0: 0, 63: 0, 67: 0}
        summary_counts = {"working_person": 0, "roaming_person": 0, 0: 0, 63: 0, 67: 0}
        timeline_logs = []
        start_time = time.time()

        pbar = tqdm(total=total_frames, desc="Streaming Video", unit="frame")
        target_frame_time = 1.0 / fps

        while cap.isOpened() and frame_idx < total_frames:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_queue.empty():
                try:
                    frame_queue.put_nowait((frame_idx, frame.copy()))
                except queue.Full:
                    pass

            try:
                inf_idx, new_boxes, new_counts = result_queue.get_nowait()
                latest_target_boxes = new_boxes
                current_counts = new_counts
                for key, count in current_counts.items():
                    if count > 0:
                        summary_counts[key] = summary_counts.get(key, 0) + 1
            except queue.Empty:
                pass

            displayed_boxes = smooth_update_boxes(displayed_boxes, latest_target_boxes, factor=0.35)

            # Draw bounding boxes (RED for Roaming Person, GREEN for Seated Working Person, CYAN for Laptop, CORAL for Phone)
            for item in displayed_boxes:
                cls_id = item["cls_id"]
                conf = item["confidence"]
                x1, y1, x2, y2 = item["bbox"]
                is_roaming = item.get("is_roaming", False)
                
                if cls_id == 0:
                    color = CLASS_COLORS["roaming_person"] if is_roaming else CLASS_COLORS["working_person"]
                    label_name = "ROAMING" if is_roaming else "PERSON"
                    thickness = 3 if is_roaming else 2
                else:
                    color = CLASS_COLORS.get(cls_id, CLASS_COLORS["default"])
                    label_name = "LAPTOP" if cls_id == 63 else "MOBILE PHONE"
                    thickness = 2

                draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness=thickness, r=8)
                
                tag_text = f"{label_name} {int(conf*100)}%"
                (text_w, text_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
                
                tag_y1 = max(y1 - text_h - 10, 0)
                tag_y2 = max(y1, text_h + 10)
                
                cv2.rectangle(frame, (x1, tag_y1), (x1 + text_w + 14, tag_y2), color, -1)
                cv2.putText(frame, tag_text, (x1 + 7, tag_y2 - 5), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255) if is_roaming else (15, 23, 42), 1, cv2.LINE_AA)

            # Status alert banner
            alert_msg = None
            roam_cnt = current_counts.get("roaming_person", 0)
            work_cnt = current_counts.get("working_person", 0)
            phone_cnt = current_counts.get(67, 0)
            
            if roam_cnt > 0:
                alert_msg = f"UNPRODUCTIVE ROAMING DETECTED ({roam_cnt} PERSONS)"
            elif phone_cnt > 0:
                alert_msg = "MOBILE PHONE USAGE DETECTED (POSSIBLE DISTRACTION)"
            elif work_cnt > 0:
                alert_msg = "PRODUCTIVE LAPTOP WORKSTATION ACTIVE"
            else:
                alert_msg = "WORKSTATION VACANT"

            draw_hud_overlay(frame, frame_idx, total_frames, fps, current_counts, alert_msg)
            out.write(frame)
            cv2.imshow("CCTV Live Monitor", frame)

            elapsed = time.time() - loop_start
            wait_time = max(1, int((target_frame_time - elapsed) * 1000))
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

            pbar.update(1)

        stop_event.set()
        frame_queue.put(None)
        worker_thread.join(timeout=2.0)

    else:
        # Standard offline video processing loop
        frame_idx = 0
        start_time = time.time()

        timeline_logs = []
        summary_counts = {"working_person": 0, "roaming_person": 0, 0: 0, 63: 0, 67: 0}
        
        prev_boxes = []
        next_boxes = []
        current_counts = {"working_person": 0, "roaming_person": 0, 0: 0, 63: 0, 67: 0}
        
        pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            sub_step = (frame_idx - 1) % stride
            
            if sub_step == 0:
                prev_boxes = next_boxes if next_boxes else []
                next_boxes, current_counts = detect_frame_objects(model, frame, conf_thresh=conf_thresh, imgsz=imgsz, use_zoom=use_zoom)
                if not prev_boxes:
                    prev_boxes = next_boxes

            for key, count in current_counts.items():
                if count > 0:
                    summary_counts[key] = summary_counts.get(key, 0) + 1

            for item in next_boxes:
                cls_id = item["cls_id"]
                conf = item["confidence"]
                x1, y1, x2, y2 = item["bbox"]
                is_roaming = item.get("is_roaming", False)
                
                if cls_id == 0:
                    color = CLASS_COLORS["roaming_person"] if is_roaming else CLASS_COLORS["working_person"]
                    label_name = "ROAMING" if is_roaming else "PERSON"
                    thickness = 3 if is_roaming else 2
                else:
                    color = CLASS_COLORS.get(cls_id, CLASS_COLORS["default"])
                    label_name = "LAPTOP" if cls_id == 63 else "MOBILE PHONE"
                    thickness = 2

                draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness=thickness, r=8)
                
                tag_text = f"{label_name} {int(conf*100)}%"
                (text_w, text_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
                
                tag_y1 = max(y1 - text_h - 10, 0)
                tag_y2 = max(y1, text_h + 10)
                
                cv2.rectangle(frame, (x1, tag_y1), (x1 + text_w + 14, tag_y2), color, -1)
                cv2.putText(frame, tag_text, (x1 + 7, tag_y2 - 5), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255) if is_roaming else (15, 23, 42), 1, cv2.LINE_AA)

            roam_cnt = current_counts.get("roaming_person", 0)
            work_cnt = current_counts.get("working_person", 0)
            phone_cnt = current_counts.get(67, 0)
            
            if roam_cnt > 0:
                alert_msg = f"UNPRODUCTIVE ROAMING DETECTED ({roam_cnt} PERSONS)"
            elif phone_cnt > 0:
                alert_msg = "MOBILE PHONE USAGE DETECTED (POSSIBLE DISTRACTION)"
            elif work_cnt > 0:
                alert_msg = "PRODUCTIVE LAPTOP WORKSTATION ACTIVE"
            else:
                alert_msg = "WORKSTATION VACANT"

            draw_hud_overlay(frame, frame_idx, total_frames, fps, current_counts, alert_msg)
            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pbar.close()

    elapsed_time = time.time() - start_time
    actual_fps = frame_idx / max(elapsed_time, 0.001)
    print(f"[SUCCESS] Video processing complete in {elapsed_time:.1f} seconds ({actual_fps:.1f} FPS processing speed).")
    
    # Save Summary Report
    summary_path = os.path.join(output_dir, "summary_report.json")
    total_sec = round(total_frames / fps, 2)
    person_frames = summary_counts.get(0, 0)
    working_frames = summary_counts.get("working_person", 0)
    roaming_frames = summary_counts.get("roaming_person", 0)
    laptop_frames = summary_counts.get(63, 0)
    phone_frames = summary_counts.get(67, 0)
    
    summary_data = {
        "video_path": input_path,
        "output_video": output_video_path,
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": total_sec,
        "duration_formatted": f"{int(total_sec//60):02d}:{int(total_sec%60):02d}",
        "processing_time_sec": round(elapsed_time, 2),
        "stride_used": stride,
        "zoom_inference_enabled": use_zoom,
        "working_person_presence": {
            "frames": working_frames,
            "seconds": round(working_frames / fps, 2),
            "percentage": round((working_frames / total_frames) * 100, 2)
        },
        "roaming_person_presence": {
            "frames": roaming_frames,
            "seconds": round(roaming_frames / fps, 2),
            "percentage": round((roaming_frames / total_frames) * 100, 2)
        },
        "laptop_presence": {
            "frames": laptop_frames,
            "seconds": round(laptop_frames / fps, 2),
            "percentage": round((laptop_frames / total_frames) * 100, 2)
        },
        "phone_presence": {
            "frames": phone_frames,
            "seconds": round(phone_frames / fps, 2),
            "percentage": round((phone_frames / total_frames) * 100, 2)
        }
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[+] Results saved:")
    print(f"    - Video: {output_video_path}")
    print(f"    - Summary: {summary_path}")
    return summary_data

def main():
    parser = argparse.ArgumentParser(description="CCTV Video Object Detection & Monitoring")
    parser.add_argument("--input", type=str, default=None, help="Path to input MP4 video")
    parser.add_argument("--output-dir", type=str, default="output_vedio", help="Output directory")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Path to YOLO model weights")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for offline processing")
    parser.add_argument("--conf", type=float, default=0.04, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit max frames for testing")
    parser.add_argument("--start-frame", type=int, default=0, help="Seek start frame")
    parser.add_argument("--show", action="store_true", help="Display OpenCV window")
    parser.add_argument("--enable-zoom", action="store_true", default=True, help="Enable zoomed tiled inference for small objects & distant people")
    parser.add_argument("--disable-zoom", action="store_false", dest="enable_zoom", help="Disable zoomed tiled inference")
    args = parser.parse_args()

    input_file = args.input
    if not input_file:
        search_pattern = os.path.join("input_vedio", "*.mp4")
        found = glob.glob(search_pattern)
        if found:
            input_file = found[0]
            print(f"[+] Auto-detected input video: {input_file}")
        else:
            sys.exit("[ERROR] No MP4 file found in input_vedio/. Please specify --input path.")

    process_video(
        input_path=input_file,
        output_dir=args.output_dir,
        model_path=args.model,
        stride=args.stride,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        show=args.show,
        use_zoom=args.enable_zoom
    )

if __name__ == "__main__":
    main()

