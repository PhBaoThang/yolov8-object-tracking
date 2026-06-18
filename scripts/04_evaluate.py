# scripts/04_evaluate.py
"""
Bước 4: Tính MOTA / MOTP / IDF1 / HOTA / ID Switch / Precision / Recall.
Pseudo Ground Truth = YOLOv8m fine-tuned output.
Output: metrics_comparison.csv
"""

import sys, os, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.metrics import extract_frame_dict, compute_mot_metrics, compute_hota

# ── Load tracking JSON ────────────────────────────────────────────────
tracking_results = {}
for key in ['yolov8n', 'yolov8s', 'yolov8m']:
    fname = f'tracking_{key}.json'
    if os.path.exists(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            tracking_results[key] = json.load(f)
        print(f"✅ {fname}  ({len(tracking_results[key]['results'])} frames)")
    else:
        print(f"❌ {fname} KHÔNG TÌM THẤY! Hãy chạy 03_tracking.py trước.")

if 'yolov8m' not in tracking_results:
    raise SystemExit("❌ Cần có tracking_yolov8m.json (Pseudo-GT)!")

# ── Pseudo GT = YOLOv8m ───────────────────────────────────────────────
gt_frames = extract_frame_dict(tracking_results['yolov8m'])
print("\n📌 Pseudo Ground Truth = YOLOv8m fine-tuned")

# ── Tính MOT metrics ──────────────────────────────────────────────────
print("\n⏳ Tính MOTA / MOTP / IDF1 / Precision / Recall / ID Switch...\n")
mot_results = {}
for key in ['yolov8n', 'yolov8s', 'yolov8m']:
    if key not in tracking_results:
        continue
    hyp = extract_frame_dict(tracking_results[key])
    print(f"  → {key}...", end=" ", flush=True)
    mot_results[key] = compute_mot_metrics(gt_frames, hyp)
    s    = mot_results[key]
    mota = s['mota'].values[0]
    idf1 = s['idf1'].values[0]
    idsw = s['num_switches'].values[0]
    print(f"MOTA={mota*100:.1f}%  IDF1={idf1*100:.1f}%  ID-SW={int(idsw)}  ✅")

# ── Tính HOTA ─────────────────────────────────────────────────────────
print("\n⏳ Tính HOTA (19 alpha × 3 model)...\n")
hota_results = {}
for key in ['yolov8n', 'yolov8s', 'yolov8m']:
    if key not in tracking_results:
        continue
    print(f"\n[{key}]")
    hyp = extract_frame_dict(tracking_results[key])
    hota_results[key] = compute_hota(gt_frames, hyp)
    print(f"  HOTA = {hota_results[key]*100:.2f}% ✅")

# ── Tổng hợp bảng ─────────────────────────────────────────────────────
def safe_get(key, metric):
    if key not in mot_results:
        return None
    try:
        v = mot_results[key][metric].values[0]
        return None if (isinstance(v, float) and np.isnan(v)) else v
    except Exception:
        return None

def fmt(v, scale=100, decimals=2):
    return f"{v * scale:.{decimals}f}" if v is not None else "N/A"

table = {}
for key in ['yolov8n', 'yolov8s', 'yolov8m']:
    mota = safe_get(key, 'mota')
    motp = safe_get(key, 'motp')
    idf1 = safe_get(key, 'idf1')
    prec = safe_get(key, 'precision')
    rec  = safe_get(key, 'recall')
    idsw = safe_get(key, 'num_switches')
    hota = hota_results.get(key)
    table[key] = {
        'MOTA (%)'     : fmt(mota),
        'MOTP (%)'     : fmt(1 - motp) if motp is not None else 'N/A',
        'IDF1 (%)'     : fmt(idf1),
        'HOTA (%)'     : fmt(hota),
        'ID Switch'    : str(int(idsw)) if idsw is not None else 'N/A',
        'Precision (%)': fmt(prec),
        'Recall (%)'   : fmt(rec),
    }

METRICS = ['MOTA (%)', 'MOTP (%)', 'IDF1 (%)', 'HOTA (%)',
           'ID Switch', 'Precision (%)', 'Recall (%)']
COL     = 17
SEP     = "─" * (22 + COL * 3)

print("\n" + "═" * (22 + COL * 3))
print("   📊 KẾT QUẢ ĐÁNH GIÁ ĐỊNH LƯỢNG — TRACKING METRICS")
print("═" * (22 + COL * 3))
print(f"\n{'Metric':<22}{'YOLOv8s':>{COL}}{'YOLOv8n':>{COL}}{'YOLOv8m':>{COL}}")
print(SEP)
for m in METRICS:
    row = f"{m:<22}"
    for key in ['yolov8s', 'yolov8n', 'yolov8m']:
        row += f"{table[key].get(m, 'N/A'):>{COL}}"
    print(row)
print(SEP)
print("""
📌 Ghi chú:
   • Pseudo Ground Truth : YOLOv8m fine-tuned
   • IoU Threshold       : 0.5
   • HOTA α range        : 0.05 → 0.95 (19 bước)
   • YOLOv8m ≈ 100% là expected (nó chính là GT reference)
""")

# ── Lưu CSV ───────────────────────────────────────────────────────────
with open('metrics_comparison.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['Metric', 'YOLOv8s', 'YOLOv8n', 'YOLOv8m'])
    for m in METRICS:
        w.writerow([m] + [table[key].get(m, 'N/A') for key in ['yolov8s', 'yolov8n', 'yolov8m']])
print("✅ Saved → metrics_comparison.csv")