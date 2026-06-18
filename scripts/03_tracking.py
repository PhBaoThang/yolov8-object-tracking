# scripts/03_tracking.py
"""
Bước 3: Chạy tracking với 3 model đã fine-tune.
Output:
  - output_yolov8n.mp4 / yolov8s / yolov8m
  - tracking_yolov8n.csv / .json  (tương tự s, m)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.tracker import YOLOv8Tracker

# ── Config ────────────────────────────────────────────────────────────
MODELS_INFO = {
    'yolov8n': {
        'name'      : 'YOLOv8 Nano (Fine-Tuned)',
        'checkpoint': 'runs/detect/yolov8n_finetuned/weights/best.pt',
    },
    'yolov8s': {
        'name'      : 'YOLOv8 Small (Fine-Tuned)',
        'checkpoint': 'runs/detect/yolov8s_finetuned/weights/best.pt',
    },
    'yolov8m': {
        'name'      : 'YOLOv8 Medium (Fine-Tuned)',
        'checkpoint': 'runs/detect/yolov8m_finetuned/weights/best.pt',
    },
}

TRACKING_CONFIG = {
    'input_video'   : 'data/apple.mp4',   # ← đặt video vào thư mục data/
    'tracker_config': 'bytetrack.yaml',
    'conf_threshold': 0.5,
    'iou_threshold' : 0.45,
    'target_classes': None,               # None = track tất cả class
}

# ── Run ───────────────────────────────────────────────────────────────
print("=" * 55)
print("  🚀 BẮT ĐẦU TRACKING — 3 MODEL")
print("=" * 55)

all_summaries = {}
for model_key, model_info in MODELS_INFO.items():
    print(f"\n{'─'*55}")
    print(f"  Model: {model_info['name']}")
    print(f"{'─'*55}")

    tracker = YOLOv8Tracker(model_key, model_info, TRACKING_CONFIG)
    summary = tracker.track_video()
    all_summaries[model_key] = summary

print("\n\n✅ Tracking xong!")
print(f"\n{'Model':<12} {'Frames':>8} {'Dets':>8} {'Unique IDs':>12} {'FPS':>8}")
print("─" * 52)
for k, s in all_summaries.items():
    print(f"{k:<12} {s['total_frames']:>8} {s['total_detections']:>8} "
          f"{s['unique_track_ids']:>12} {s['proc_fps']:>8.1f}")