# CCTV-Monitoring

An AI-powered CCTV Video Analytics and Security Compliance System designed for multi-camera smart surveillance, biometric compliance tracking, productivity analysis, and vehicle detection.

## 🚀 Key Modules

### 1. 🖐️ Biometric Attendance Compliance Detection (`fingurprint_detect/`)
- **Direction-Aware Tracking**: Automatically determines whether individuals are `ENTERING` or `LEAVING`.
- **YOLO-Pose AI Hand Tracking**: Tracks skeleton keypoints (wrists & fingertips) in real time to verify physical interaction with biometric fingerprint scanners.
- **Graceful Entry Verification**: Allows people to casually scan upon entering; flags violations only when someone fully enters without biometric compliance.
- **Automated Violation Alerts**: Captures clear visual snapshots of unpunched entries and instantly dispatches email alerts via SMTP.

### 2. 📊 Workplace Productivity Analysis (`productiveity_analysis/`)
- Real-time person detection, workstation tracking, and seating occupancy analysis.
- Generates detailed engagement analytics and logging over RTSP camera streams.

### 3. 🚗 Vehicle Detection & Parking Analytics (`vehical_detection/`)
- Real-time vehicle detection, tracking, and traffic monitoring.

### 4. 📹 Multi-Camera RTSP Stream Engine (`camera.py`, `lanconcamera.py`)
- Low-latency asynchronous RTSP capture pipeline with automatic reconnection over TCP.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- OpenCV & Ultralytics YOLO

### Installation
```bash
pip install ultralytics opencv-python numpy python-dotenv
```

### Configuration
Create a `.env` file in the root directory:
```env
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your_password
CAMERA_PORT=554
CAMERA1_IP=192.168.90.20
CAMERA2_IP=192.168.90.21

# Email Alerts
EMAIL_SENDER=your_email@gmail.com
EMAIL_APP_PASSWORD=your_app_password
EMAIL_RECIPIENTS=manager@example.com
```

### Running the Attendance Compliance Detector
```bash
cd fingurprint_detect
python live_fingerprint_attendance_detector.py --cam 1
```

---

## 🔒 Security Note
Never commit `.env` or sensitive camera credentials to version control.
