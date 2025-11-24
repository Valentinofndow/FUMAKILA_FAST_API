# Bottle Detection API

## How to Run
1. pip install -r requirements.txt
2. uvicorn server:app --reload

## Endpoints
- GET /health → cek apakah API jalan
- POST /predict → kirim gambar, return prediction + confidence + status

## Model
YOLOv11s trained on 3 classes: cap_on, cap_off_wick_ok, cap_off_wick_ng
