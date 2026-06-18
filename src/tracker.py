# src/tracker.py
"""
YOLOv8Tracker — chạy tracking cho 1 model, lưu CSV + JSON + video.
"""

import cv2
import csv
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


class YOLOv8Tracker:
    """Wrapper gọn gàng cho YOLO + ByteTrack."""

    def __init__(self, model_key: str, model_info: dict, tracking_config: dict):
        """
        Parameters
        ----------
        model_key       : 'yolov8n' | 'yolov8s' | 'yolov8m'
        model_info      : {'name': str, 'checkpoint': str}
        tracking_config : xem TRACKING_CONFIG bên dưới
        """
        self.model_key       = model_key
        self.model_info      = model_info
        self.tracking_config = tracking_config
        self.model           = YOLO(model_info['checkpoint'])
        print(f"✅ Loaded: {model_info['name']}  ({model_info['checkpoint']})")

    # ------------------------------------------------------------------
    def _get_color(self, track_id: int):
        np.random.seed(int(track_id) % 1000)
        return tuple(np.random.randint(50, 220, 3).tolist())

    # ------------------------------------------------------------------
    def track_video(self) -> dict:
        """
        Chạy tracking toàn bộ video.

        Returns
        -------
        dict  — summary thống kê (total frames, FPS, paths, ...)
        """
        key        = self.model_key
        cfg        = self.tracking_config
        input_path = cfg['input_video']
        out_mp4    = f'output_{key}.mp4'
        out_csv    = f'tracking_{key}.csv'
        out_json   = f'tracking_{key}.json'

        cap       = cv2.VideoCapture(input_path)
        fps       = cap.get(cv2.CAP_PROP_FPS)
        width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  📹 {width}x{height} @ {fps:.1f}fps | {total_frm} frames")

        writer = cv2.VideoWriter(
            out_mp4,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (width, height)
        )

        tracking_data = []
        frame_count   = 0
        start_time    = datetime.now()
        header_text   = (f"{self.model_info['name']} | "
                         f"conf={cfg['conf_threshold']} | "
                         f"{cfg['tracker_config']}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results = self.model.track(
                frame,
                persist=True,
                tracker=cfg['tracker_config'],
                conf=cfg['conf_threshold'],
                iou=cfg['iou_threshold'],
                classes=cfg['target_classes'],
                verbose=False,
            )

            annotated  = frame.copy()
            frame_dets = []

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes     = results[0].boxes.xyxy.cpu().numpy()
                confs     = results[0].boxes.conf.cpu().numpy()
                cls_ids   = results[0].boxes.cls.cpu().numpy().astype(int)
                track_ids = (results[0].boxes.id.int().cpu().numpy().tolist()
                             if results[0].boxes.id is not None
                             else [None] * len(boxes))

                for box, tid, conf, cid in zip(boxes, track_ids, confs, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cls_name = self.model.names[cid]
                    color    = self._get_color(tid) if tid is not None else (0, 255, 0)
                    label    = (f"ID:{tid} {cls_name} {conf:.2f}" if tid
                                else f"{cls_name} {conf:.2f}")

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    frame_dets.append({
                        'track_id'  : int(tid) if tid is not None else -1,
                        'bbox'      : [x1, y1, x2, y2],
                        'confidence': round(float(conf), 4),
                        'class_id'  : int(cid),
                        'class_name': cls_name,
                    })

            # Watermark
            cv2.rectangle(annotated, (0, 0), (width, 28), (0, 0, 0), -1)
            cv2.putText(annotated, header_text, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(annotated, f"Frame:{frame_count}/{total_frm} | Det:{len(frame_dets)}",
                        (width - 300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            writer.write(annotated)
            tracking_data.append({
                'frame_id'  : frame_count,
                'timestamp' : round(frame_count / fps, 3),
                'detections': frame_dets,
            })

            if frame_count % 100 == 0:
                elapsed = (datetime.now() - start_time).seconds
                print(f"  ⏳ Frame {frame_count}/{total_frm} | {elapsed}s elapsed")

        cap.release()
        writer.release()
        elapsed_total = max(1, (datetime.now() - start_time).seconds)

        # ── Lưu CSV ───────────────────────────────────────────────────
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['frame_id', 'timestamp', 'track_id', 'class_name',
                        'confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
            for fr in tracking_data:
                for d in fr['detections']:
                    b = d['bbox']
                    w.writerow([fr['frame_id'], fr['timestamp'], d['track_id'],
                                d['class_name'], d['confidence'],
                                b[0], b[1], b[2], b[3]])

        # ── Lưu JSON ──────────────────────────────────────────────────
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({
                'model_key'      : key,
                'model_name'     : self.model_info['name'],
                'checkpoint'     : self.model_info['checkpoint'],
                'tracking_config': cfg,
                'video_info'     : {'width': width, 'height': height,
                                    'fps': fps, 'total_frames': total_frm},
                'results'        : tracking_data,
            }, f, indent=2, ensure_ascii=False)

        total_dets = sum(len(f['detections']) for f in tracking_data)
        all_ids    = {d['track_id'] for f in tracking_data
                      for d in f['detections'] if d['track_id'] != -1}

        summary = {
            'model_key'         : key,
            'model_name'        : self.model_info['name'],
            'total_frames'      : frame_count,
            'total_detections'  : total_dets,
            'unique_track_ids'  : len(all_ids),
            'avg_det_per_frame' : round(total_dets / frame_count, 2) if frame_count else 0,
            'processing_time_s' : elapsed_total,
            'proc_fps'          : round(frame_count / elapsed_total, 1),
            'output_video'      : out_mp4,
            'output_csv'        : out_csv,
            'output_json'       : out_json,
        }
        print(f"  ✅ Xong! {elapsed_total}s | {total_dets} dets | {len(all_ids)} unique IDs")
        return summary