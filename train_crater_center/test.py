"""
Evaluation for CraterShuffleNet.

Predictions are decoded by applying a confidence threshold then
suppressing duplicates with a fast circular-distance NMS.
Matching is done with circular IoU (two-circle intersection area).
"""

import torch, math, numpy as np
from typing import List


# ── Circular IoU ─────────────────────────────────────────────────────────────
def _circle_iou(cx1, cy1, r1, cx2, cy2, r2):
    d = math.hypot(cx1 - cx2, cy1 - cy2)
    if d >= r1 + r2 or r1 <= 0 or r2 <= 0:
        return 0.0
    if d <= abs(r1 - r2):
        sr, lr = min(r1, r2), max(r1, r2)
        return (sr / lr) ** 2
    alpha = 2 * math.acos(max(-1.0, min(1.0,
                (d*d + r1*r1 - r2*r2) / (2 * d * r1))))
    beta  = 2 * math.acos(max(-1.0, min(1.0,
                (d*d + r2*r2 - r1*r1) / (2 * d * r2))))
    inter = 0.5*r1**2*(alpha - math.sin(alpha)) + 0.5*r2**2*(beta - math.sin(beta))
    union = math.pi*(r1**2 + r2**2) - inter
    return inter / union if union > 0 else 0.0


def _circle_nms(detections, iou_thresh=0.3):
    """detections: list of (cx, cy, r, score) — returns kept list."""
    detections = sorted(detections, key=lambda x: -x[3])
    keep, suppressed = [], set()
    for i, d in enumerate(detections):
        if i in suppressed:
            continue
        keep.append(d)
        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            if _circle_iou(d[0], d[1], d[2], detections[j][0],
                           detections[j][1], detections[j][2]) > iou_thresh:
                suppressed.add(j)
    return keep


# ── Decode predictions ────────────────────────────────────────────────────────
def decode_predictions(preds, logits,
                       conf_thresh=0.5, nms_iou_thresh=0.3):
    """
    preds  : (N_MAX, 3)  sigmoid [cx, cy, r]     — single image
    logits : (N_MAX,)    raw logits               — single image
    Returns list of (cx_norm, cy_norm, r_norm, score)
    """
    scores = torch.sigmoid(logits)
    dets = []
    for i in range(preds.shape[0]):
        s = float(scores[i])
        if s >= conf_thresh:
            cx, cy, r = preds[i].tolist()
            dets.append((cx, cy, r, s))
    return _circle_nms(dets, nms_iou_thresh)


# ── Matching ──────────────────────────────────────────────────────────────────
def match_craters(pred_list, gt_list, iou_thresh=0.3):
    """
    pred_list : [(cx, cy, r, score), ...]  already sorted by score desc
    gt_list   : [(cx, cy, r), ...]
    Returns (tp, fp, fn, list_of_|r_pred - r_gt|)
    """
    if not pred_list:
        return 0, 0, len(gt_list), []
    if not gt_list:
        return 0, len(pred_list), 0, []

    matched_gt = set()
    matched_pr = set()
    rad_errors = []

    for i, (px, py, pr, ps) in enumerate(pred_list):
        best_iou, best_j = iou_thresh, -1
        for j, (gx, gy, gr) in enumerate(gt_list):
            if j in matched_gt:
                continue
            iou = _circle_iou(px, py, pr, gx, gy, gr)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched_gt.add(best_j)
            matched_pr.add(i)
            rad_errors.append(abs(pr - gt_list[best_j][2]))

    tp = len(matched_pr)
    return tp, len(pred_list) - tp, len(gt_list) - tp, rad_errors


# ── Batch metric computation ──────────────────────────────────────────────────
def compute_metrics(preds_batch, logits_batch, gt_boxes_batch, mask_batch,
                    conf_thresh=0.5, nms_iou_thresh=0.3, match_iou_thresh=0.3):
    """
    All inputs: CPU tensors.
    Returns dict: precision, recall, f1, radius_mae_norm, tp, fp, fn
    """
    B = preds_batch.shape[0]
    total_tp = total_fp = total_fn = 0
    all_rad_err = []

    for i in range(B):
        dets = decode_predictions(preds_batch[i], logits_batch[i],
                                  conf_thresh, nms_iou_thresh)
        n_gt = int(mask_batch[i].sum())
        gt_list = [tuple(gt_boxes_batch[i, j].tolist()) for j in range(n_gt)]

        tp, fp, fn, errs = match_craters(dets, gt_list, match_iou_thresh)
        total_tp += tp; total_fp += fp; total_fn += fn
        all_rad_err.extend(errs)

    prec = total_tp / max(total_tp + total_fp, 1)
    rec  = total_tp / max(total_tp + total_fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-6)
    mae  = float(np.mean(all_rad_err)) if all_rad_err else 0.0

    return {"precision": prec, "recall": rec, "f1": f1,
            "radius_mae_norm": mae,
            "tp": total_tp, "fp": total_fp, "fn": total_fn}
