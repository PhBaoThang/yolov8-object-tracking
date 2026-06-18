# src/metrics.py
"""
Tính toán các tracking metrics:
  - MOTA, MOTP, IDF1, Precision, Recall, ID Switch  (qua motmetrics)
  - HOTA  (custom — Luiten et al., IJCV 2021)
"""

import numpy as np
import motmetrics as mm
from scipy.optimize import linear_sum_assignment

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


# ──────────────────────────────────────────────────────────────────────
def extract_frame_dict(data: dict) -> dict:
    """JSON tracking result → {frame_id: [detections]}"""
    return {fr['frame_id']: fr['detections'] for fr in data['results']}


# ──────────────────────────────────────────────────────────────────────
def compute_iou(boxA: list, boxB: list) -> float:
    """IoU giữa 2 bounding box [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-8)


# ──────────────────────────────────────────────────────────────────────
def compute_mot_metrics(gt_frames: dict, hyp_frames: dict,
                        iou_threshold: float = 0.5):
    """
    Tính MOTA, MOTP, IDF1, Precision, Recall, ID Switch.

    Parameters
    ----------
    gt_frames  : {frame_id: [det_dict]}  — ground truth
    hyp_frames : {frame_id: [det_dict]}  — hypothesis (tracker output)
    iou_threshold : float, ngưỡng IoU để coi là match

    Returns
    -------
    pandas.DataFrame  — motmetrics summary
    """
    acc        = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_frames.keys()) | set(hyp_frames.keys()))

    for fid in all_frames:
        gt_dets  = gt_frames.get(fid, [])
        hyp_dets = hyp_frames.get(fid, [])
        gt_ids   = [d['track_id'] for d in gt_dets]
        hyp_ids  = [d['track_id'] for d in hyp_dets]
        n_gt, n_hyp = len(gt_dets), len(hyp_dets)

        if n_gt == 0 and n_hyp == 0:
            acc.update([], [], np.empty((0, 0)))
            continue

        dist = np.full((n_gt, n_hyp), np.nan)
        for i, gd in enumerate(gt_dets):
            for j, hd in enumerate(hyp_dets):
                iou = compute_iou(gd['bbox'], hd['bbox'])
                if iou >= iou_threshold:
                    dist[i][j] = 1.0 - iou

        acc.update(gt_ids, hyp_ids, dist)

    mh      = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'mota', 'motp', 'idf1',
        'precision', 'recall', 'num_switches',
        'num_matches', 'num_misses', 'num_false_positives',
        'num_unique_objects', 'num_detections'
    ], name='result')
    return summary


# ──────────────────────────────────────────────────────────────────────
def compute_hota(gt_frames: dict, hyp_frames: dict,
                 alphas: np.ndarray = None) -> float:
    """
    HOTA = mean_α { sqrt(DetA(α) × AssA(α)) }
    Ref: Luiten et al., IJCV 2021

    Parameters
    ----------
    gt_frames  : {frame_id: [det_dict]}
    hyp_frames : {frame_id: [det_dict]}
    alphas     : array of IoU thresholds (default: 0.05→0.95, step 0.05)

    Returns
    -------
    float — HOTA score (0–1)
    """
    if alphas is None:
        alphas = np.arange(0.05, 0.96, 0.05)   # 19 ngưỡng

    gt_traj_len, hyp_traj_len = {}, {}
    for dets in gt_frames.values():
        for d in dets:
            gt_traj_len[d['track_id']] = gt_traj_len.get(d['track_id'], 0) + 1
    for dets in hyp_frames.values():
        for d in dets:
            hyp_traj_len[d['track_id']] = hyp_traj_len.get(d['track_id'], 0) + 1

    all_frames     = sorted(set(gt_frames.keys()) | set(hyp_frames.keys()))
    hota_per_alpha = []

    for alpha in tqdm(alphas, desc="  HOTA α", leave=True):
        matches     = []
        tp = fn = fp = 0

        for fid in all_frames:
            gt_dets  = gt_frames.get(fid, [])
            hyp_dets = hyp_frames.get(fid, [])

            if len(gt_dets) == 0:
                fp += len(hyp_dets);  continue
            if len(hyp_dets) == 0:
                fn += len(gt_dets);   continue

            iou_mat = np.zeros((len(gt_dets), len(hyp_dets)))
            for i, gd in enumerate(gt_dets):
                for j, hd in enumerate(hyp_dets):
                    iou_mat[i][j] = compute_iou(gd['bbox'], hd['bbox'])

            r_ind, c_ind       = linear_sum_assignment(-iou_mat)
            matched_gt, matched_hyp = set(), set()

            for r, c in zip(r_ind, c_ind):
                if iou_mat[r][c] >= alpha:
                    tp += 1
                    matched_gt.add(r);  matched_hyp.add(c)
                    matches.append((gt_dets[r]['track_id'], hyp_dets[c]['track_id']))

            fn += len(gt_dets)  - len(matched_gt)
            fp += len(hyp_dets) - len(matched_hyp)

        deta = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
        if tp == 0:
            hota_per_alpha.append(0.0);  continue

        tpa_count = {}
        for (g, h) in matches:
            tpa_count[(g, h)] = tpa_count.get((g, h), 0) + 1

        assa_sum = 0.0
        for (g, h) in matches:
            tpa   = tpa_count[(g, h)]
            fna   = gt_traj_len.get(g, 0)  - tpa
            fpa   = hyp_traj_len.get(h, 0) - tpa
            denom = tpa + fna + fpa
            assa_sum += (tpa / denom) if denom > 0 else 0.0

        assa = assa_sum / tp
        hota_per_alpha.append(np.sqrt(deta * assa))

    return float(np.mean(hota_per_alpha))