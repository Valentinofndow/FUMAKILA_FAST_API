from ultralytics import YOLO
import cv2

# Load model hasil training
model = YOLO("model/runs/detect/train/weights/best.pt")  # Pastikan path bener

# Buka webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi
    results = model.predict(frame, conf=0.7, verbose=False)

    # Annotasi frame
    annotated_frame = results[0].plot()

    # Tampilkan
    cv2.imshow("YOLOv11 Webcam Detection", annotated_frame)

    # ESC buat keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
