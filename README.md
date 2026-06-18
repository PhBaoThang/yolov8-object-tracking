# 📦 YOLOv8 Object Tracking – Băng Chuyền Nhà Máy

So sánh hiệu năng tracking của **YOLOv8n / YOLOv8s / YOLOv8m** (fine-tuned)
trên video thùng hộp di chuyển trên băng chuyền nhà máy, sử dụng ByteTrack.

---

## 📁 Cấu trúc repo

```
boxes-tracking-conveyor/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── tracker.py          # YOLOv8Tracker class
│   └── metrics.py          # MOTA/MOTP/IDF1/HOTA
├── scripts/
│   ├── 01_download_dataset.py
│   ├── 02_finetune.py
│   ├── 03_tracking.py
│   └── 04_evaluate.py
└── data/
    └── .gitkeep            # Đặt video băng chuyền vào đây
```

---

## ⚙️ Yêu cầu môi trường

- Python 3.10+
- Google Colab (khuyến nghị) hoặc local GPU
- CUDA không bắt buộc (CPU chạy được nhưng chậm hơn)
- Đã test trên Google Colab (T4 GPU)

---

## 🚀 Hướng dẫn tái tạo kết quả

### Bước 0 — Clone repo & cài thư viện

```bash
git clone https://github.com/PhBaoThang/yolov8-object-tracking
cd yolov8-object-tracking
pip install -r requirements.txt
```

### Bước 1 — Chuẩn bị video

Đặt file video vào thư mục `data/`:

```
data/boxes.mp4
```

### Bước 2 — Download & chuẩn bị dataset

```bash
python scripts/01_download_dataset.py
```

Tự động download từ Roboflow, giải nén và split train/valid (80/20).

### Bước 3 — Fine-tune 3 model

```bash
python scripts/02_finetune.py
```

Training 50 epochs, freeze 10 lớp đầu.
Weights lưu tại:

```
runs/detect/yolov8n_finetuned/weights/best.pt
runs/detect/yolov8s_finetuned/weights/best.pt
runs/detect/yolov8m_finetuned/weights/best.pt
```

### Bước 4 — Chạy tracking

```bash
python scripts/03_tracking.py
```

Output:

```
output_yolov8n.mp4   tracking_yolov8n.csv   tracking_yolov8n.json
output_yolov8s.mp4   tracking_yolov8s.csv   tracking_yolov8s.json
output_yolov8m.mp4   tracking_yolov8m.csv   tracking_yolov8m.json
```

### Bước 5 — Đánh giá định lượng

```bash
python scripts/04_evaluate.py
```

Output: `metrics_comparison.csv` và bảng so sánh in ra terminal:

```
Metric           YOLOv8n     YOLOv8s     YOLOv8m
MOTA (%)          ...         ...         ...
MOTP (%)          ...         ...         ...
IDF1 (%)          ...         ...         ...
HOTA (%)          ...         ...         ...
```

---

## 📊 Kết quả

| Metric        | YOLOv8n | YOLOv8s | YOLOv8m |
|---------------|---------|---------|---------|
| MOTA (%)      |   1.50  |  82.98  | 100.00  |
| MOTP (%)      |  86.35  |  82.43  | 100.00  |
| IDF1 (%)      |   2.73  |  91.61  | 100.00  |
| HOTA (%)      |   4.01  |  72.94  | 100.00  |
| ID Switches   |   5     |   2     |   0     |
| Precision (%) | 100.00  |  87.74  | 100.00  |
| Recall (%)    |   1.63  |  96.52  | 100.00  |

> ⚠️ YOLOv8m ≈ 100% là **expected** — vì nó được dùng làm **Pseudo Ground Truth**.  
> YOLOv8s cho kết quả tốt nhất trong 2 model còn lại với MOTA ~83%, IDF1 ~92%, Recall ~97%.  
> YOLOv8n cho kết quả kém nhất, MOTA chỉ đạt 1.50% do Recall cực thấp (~1.63%).

---

## 📌 Ghi chú kỹ thuật

- **Tracker**: ByteTrack (`bytetrack.yaml` — built-in của Ultralytics)
- **Pseudo GT**: YOLOv8m fine-tuned output (không có ground truth annotation cho video thực tế)
- **IoU Threshold**: 0.5
- **HOTA α range**: 0.05 → 0.95 (19 bước)

---

## 📦 Dataset

- **Source**: Roboflow — Box Detection Dataset
- **Split**: 80% train / 20% valid (random seed = 42)