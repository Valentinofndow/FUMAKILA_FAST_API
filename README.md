# Bottle Inspection API

Sistem ini adalah API berbasis FastAPI untuk inspeksi kualitas botol menggunakan model YOLOv11.  
API menyediakan fitur live streaming kamera, prediksi snapshot, logging hasil prediksi, dan endpoint tambahan seperti stop kamera dan pembuatan laporan (optional).

---

## 🚀 Fitur Utama

| Fitur                | Deskripsi                                                             |
|----------------------|-----------------------------------------------------------------------|
| `/frame`             | Live streaming kamera (MJPEG)                                         |
| `/predict`           | Ambil 1 frame dari kamera dan lakukan deteksi object menggunakan YOLO |
| `/stop`              | Stop kamera dan release resource                                      |
| `/health`            | Cek status API, model, dan kamera                                     |
| Logging otomatis     | Setiap hasil prediksi dicatat ke `logs.csv`                           |
| (Optional) `/report` | Generate PDF berisi ringkasan hasil deteksi                           |

---

## 📦 Instalasi

Pastikan Python minimal **3.10**.

1. Clone repository
2. Install dependencies:
    pip install -r requirements.txt
3. Jalankan server:
    uvicorn server:app --reload

---

## 📍 Endpoint Dokumentasi

| Method | Endpoint   | Deskripsi                                    |
|--------|------------|----------------------------------------------|
| GET    | `/health`  | Mengecek kondisi sistem (model dan kamera)   |
| GET    | `/frame`   | Live feed webcam                             |
| GET    | `/predict` | Capture snapshot dan lakukan inferensi model |
| GET    | `/stop`    | Stop kamera dan release resource             |
| POST   | `/report`  | Generate PDF (opsional)                      |

---