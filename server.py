from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import time
import asyncio
from ultralytics import YOLO
from datetime import datetime
import csv
import os

app = FastAPI()

# ======== Global State ========
cap = None
camera_active = False
LOG_FILE = "logs.csv"

# ======== Load Model Sekali ========
try:
    model = YOLO("model/runs/detect/train/weights/best.pt")
    print("Model loaded successfully.")
except Exception as e:
    print("ERROR loading model:", e)
    model = None


# ========== Helper: Logging ==========
def _write_log(label, confidence):
    """Append prediction log to CSV."""
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        # write header first time
        if not file_exists:
            writer.writerow(["timestamp", "prediction", "confidence"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            confidence if confidence is not None else ""
        ])


# ========== Helper: Camera ==========
def _init_camera():
    """Initialize webcam once."""
    global cap, camera_active

    if cap is None:
        print("🎥 Initializing camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera_active = True


def _read_frame():
    """Read a single frame from camera."""
    global cap
    ret, frame = cap.read()
    return frame if ret else None


def _encode_frame(frame):
    """Convert frame to JPEG format."""
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if ret else None


def _format_stream_chunk(jpeg_bytes):
    """Format response chunk in multipart streaming format."""
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + jpeg_bytes +
        b"\r\n"
    )


# ================= /HEALTH =================
@app.get("/health")
async def health_check():
    """Check system readiness"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "camera_ready": cap is not None and cap.isOpened()
    }


# ================= /FRAME =================
@app.get("/frame")
async def stream_camera():
    """Stream live webcam feed."""
    global cap, camera_active

    async def frame_generator():
        _init_camera()

        while camera_active:
            frame = _read_frame()
            if frame is None:
                break

            jpeg = _encode_frame(frame)
            if jpeg is None:
                continue

            yield _format_stream_chunk(jpeg)
            await asyncio.sleep(0.03)  # ~30 FPS

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ================= /STOP =================
@app.get("/stop")
async def stop_camera():
    """Stop camera stream and release resource."""
    global cap, camera_active

    camera_active = False
    if cap is not None:
        cap.release()
        cap = None

    return {"status": "camera stopped"}


# ================= /PREDICT =================
@app.get("/predict")
async def predict_snapshot():
    """Capture single frame and run detection"""
    global cap

    if cap is None:
        return {"error": "Camera not initialized. Open /frame first."}

    ret, frame = cap.read()
    if not ret:
        return {"error": "Failed to capture frame"}

    result = model(frame)[0]

    if len(result.boxes) == 0:
        _write_log("no_object_detected", None)
        return {
            "prediction": "no_object_detected",
            "confidence": None,
            "status": "UNDEFINED"
        }

    cls_id = int(result.boxes.cls[0])
    conf = float(result.boxes.conf[0])
    label = result.names[cls_id]

    status = "PASS" if label == "Cap_On" else "REJECT"

    # log prediction
    _write_log(label, round(conf, 3))

    return {
        "prediction": label,
        "confidence": round(conf, 3),
        "status": status
    }
