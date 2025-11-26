from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import numpy as np
import cv2
import time
import asyncio
from ultralytics import YOLO
from datetime import datetime
import csv
import os
import json
import threading

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== Global State ========
cap = None
camera_active = False
LOG_FILE = "logs.csv"
CONFIG_FILE = "config.json"
log_lock = threading.Lock()

# Counters
total_scanned = 0
total_good = 0
total_defect = 0

# Label yang dianggap GOOD
PASS_LABELS = ["Cap_On"]

# ======== Load Config (PASS LABELS) ========
def _load_config():
    """Load PASS labels from config.json or create default."""
    default_config = {"pass_labels": ["Cap_On"]}

    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump(default_config, f, indent=4)
        return default_config

    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


CONFIG = _load_config()
PASS_LABELS = CONFIG["pass_labels"]


# ======== Load Model ========
try:
    model = YOLO("model/runs/detect/train/weights/best.pt")
    print("Model loaded successfully.")
except Exception as e:
    print("ERROR loading model:", e)
    model = None


# ======== Load Counters From CSV ========
def _load_counters_from_csv():
    global total_scanned, total_good, total_defect

    if not os.path.exists(LOG_FILE):
        return

    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_scanned = len(rows)
    total_good = sum(1 for r in rows if r["prediction"] in PASS_LABELS)
    total_defect = total_scanned - total_good


_load_counters_from_csv()


# ========== Logging Helper ==========
def _write_log(label, confidence):
    with log_lock:
        file_exists = os.path.isfile(LOG_FILE)

        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["timestamp", "prediction", "confidence"])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                label,
                confidence if confidence is not None else ""
            ])


# ========== Camera Helper ==========
def _init_camera():
    global cap, camera_active

    if cap is None:
        print("🎥 Initializing camera...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera_active = True


def _read_frame():
    global cap
    ret, frame = cap.read()
    return frame if ret else None


def _encode_frame(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if ret else None


def _format_stream_chunk(jpeg_bytes):
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + jpeg_bytes +
        b"\r\n"
    )

# ==========================================
@app.get("/")
async def main():
    return {"message": "Hello World"}

# ================= /HEALTH =================
@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "camera_ready": cap is not None and cap.isOpened(),
        "total_scanned": total_scanned,
        "total_good": total_good,
        "total_defect": total_defect,
        "error_rate": round((total_defect / total_scanned) * 100, 2) if total_scanned > 0 else 0
    }


# ================= /FRAME =================
@app.get("/frame")
async def stream_camera():
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
            await asyncio.sleep(0.03)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ================= /STOP =================
@app.get("/stop") 
async def stop_camera(): 
    global cap, camera_active 
    
    camera_active = False 
    if cap is not None: 
        cap.release() 
        cap = None 
        return {"status": "camera stopped"}


# ================= /PREDICT =================
@app.get("/predict")
async def predict_snapshot():
    global cap, total_scanned, total_good, total_defect

    if cap is None:
        return {"error": "Camera not initialized. Open /frame first."}

    frame = _read_frame()
    if frame is None:
        return {"error": "Failed to capture frame"}

    result = model(frame)[0]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # =======================
    # CASE 1: NO DETECTION (UNDEFINED)
    # =======================
    if len(result.boxes) == 0:
        _write_log("no_object_detected", None)

        return {
            "timestamp": timestamp,
            "prediction": "no_object_detected",
            "confidence": None,
            "status": "UNDEFINED",
            "total_scanned": total_scanned,
            "total_good": total_good,
            "total_defect": total_defect
        }

    # =======================
    # CASE 2: ADA DETEKSI
    # =======================
    cls_id = int(result.boxes.cls[0])
    conf = float(result.boxes.conf[0])
    label = result.names[cls_id]
    confidence_rounded = round(conf, 3)

    # Update counters
    total_scanned += 1

    if label in PASS_LABELS:
        status = "PASS"
        total_good += 1
    else:
        status = "REJECT"
        total_defect += 1

    # Tulis CSV
    _write_log(label, confidence_rounded)

    return {
        "timestamp": timestamp,
        "prediction": label,
        "confidence": confidence_rounded,
        "status": status,
        "total_scanned": total_scanned,
        "total_good": total_good,
        "total_defect": total_defect
    }


# ================= /RESULT =================
@app.get("/result")
async def get_result():
    if total_scanned == 0:
        success_rate = 0
        error_rate = 0
    else:
        success_rate = round((total_good / total_scanned) * 100, 2)
        error_rate = round((total_defect / total_scanned) * 100, 2)

    return {
        "total_scanned": total_scanned,
        "total_good": total_good,
        "total_defect": total_defect,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ================= /RESET =================
@app.get("/reset")
async def reset_data():
    global total_scanned, total_good, total_defect

    total_scanned = 0
    total_good = 0
    total_defect = 0

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "prediction", "confidence"])

    return {
        "status": "reset_success",
        "message": "All counters and logs have been reset."
    }


# ================= /REPORT =================
@app.post("/report")
def generate_report():

    if not os.path.exists(LOG_FILE):
        return {"error": "No scan data found. Scan first before generating report."}

    with open(LOG_FILE, "r") as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 0:
        return {"error": "Log file empty"}

    total = len(rows)
    good = sum(1 for r in rows if r["prediction"] in PASS_LABELS)
    defects = total - good
    success_rate = round((good / total) * 100, 2)
    error_rate = round((defects / total) * 100, 2)

    rejected_rows = [r for r in rows if r["prediction"] not in PASS_LABELS]

    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)

    c.setFont("Helvetica-Bold", 20)
    c.drawString(150, 800, "FUMAKILA Bottle Inspection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 760, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 720, "Summary")

    c.setFont("Helvetica", 12)
    c.drawString(50, 690, f"Total Bottles: {total}")
    c.drawString(50, 670, f"Total Good: {good}")
    c.drawString(50, 650, f"Total Defects: {defects}")
    c.drawString(50, 630, f"Success Rate: {success_rate}%")
    c.drawString(50, 610, f"Error Rate: {error_rate}%")

    c.showPage()

    if len(rejected_rows) > 0:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(180, 800, "Defective Items Report")

        y = 760
        for idx, row in enumerate(rejected_rows, start=1):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, f"#{idx} — Prediction: {row['prediction']} | Confidence: {row['confidence']}")
            y -= 20

            if y < 100:
                c.showPage()
                y = 780
    else:
        c.setFont("Helvetica", 14)
        c.drawString(200, 700, "No Defects Detected")

    c.save()

    return FileResponse(filename, media_type="application/pdf", filename=filename)
