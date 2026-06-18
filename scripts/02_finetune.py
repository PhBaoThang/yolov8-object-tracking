# scripts/02_finetune.py
"""
Bước 2: Fine-tune YOLOv8n / YOLOv8s / YOLOv8m trên dataset Roboflow.
Kết quả weights lưu tại: runs/detect/<model>_finetuned/weights/best.pt
"""

import os
from ultralytics import YOLO

DATA_YAML   = 'roboflow_dataset/data.yaml'
EPOCHS      = 50
IMG_SIZE    = 640
BATCH       = 16
FREEZE      = 10   # đóng băng 10 lớp đầu (transfer learning)
PATIENCE    = 10

MODELS = {
    'yolov8n': 'yolov8n.pt',
    'yolov8s': 'yolov8s.pt',
    'yolov8m': 'yolov8m.pt',
}

fine_tuned_paths = {}

for model_name, pretrained_path in MODELS.items():
    print(f"\n{'='*55}")
    print(f"  🔧 Fine-tuning: {model_name}")
    print(f"{'='*55}")

    model = YOLO(pretrained_path)

    model.train(
        data     = DATA_YAML,
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        freeze   = FREEZE,
        patience = PATIENCE,
        name     = f'{model_name}_finetuned',
        exist_ok = True,
    )

    best_path = f'runs/detect/{model_name}_finetuned/weights/best.pt'
    fine_tuned_paths[model_name] = best_path

    if os.path.exists(best_path):
        print(f"  ✅ Saved → {best_path}")
    else:
        print(f"  ❌ Không tìm thấy {best_path}!")

print("\n\n✅ Fine-tune cả 3 model xong!")
for k, v in fine_tuned_paths.items():
    print(f"  {k}: {v}")