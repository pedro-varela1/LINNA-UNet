"""
Evaluation metrics for CraterCenterNet.

Detection pipeline:
  1. Threshold heatmap at `hm_thresh`.
  2. Apply 2-D non-maximum suppression (max-pool peak extraction).
  3. Extract the predicted radius at each peak location.
  4. Match predicted craters to ground-truth using a distance criterion.
  5. Compute Precision, Recall, F1 and radius MAE (km).

Key functions
-------------
  extract_peaks(heatmap, radmap, hm_thresh, nms_pool)
      → list of (cx_out, cy_out, r_norm)

  match_craters(preds, gts, max_dist_out)
      → (tp, fp, fn, radius_errors_km)

  compute_metrics(pred_hm, pred_rad, gt_hm, gt_rad, regmask, ...)
      → dict with precision, recall, f1, radius_mae_km
"""

from dataload import RADIUS_MIN_KM, RADIUS_MAX_KM, OUT_SIZE
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------
def extract_peaks(
    heatmap:  torch.Tensor,   # (H, W)  — single-sample, already on CPU
    radmap:   torch.Tensor,   # (H, W)
    hm_thresh: float = 0.3,
    nms_pool:  int   = 5,
) -> List[Tuple[int, int, float]]:
    """
    Returns list of (cx_out, cy_out, r_norm) for each detected crater.

    NMS: a pixel is a peak only if it equals the local maximum in a
    (nms_pool × nms_pool) neighbourhood.
    """
    hm = heatmap.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Max-pool NMS
    maxpool = F.max_pool2d(hm, kernel_size=nms_pool,
                           stride=1, padding=nms_pool // 2)
    keep = (hm == maxpool) & (hm >= hm_thresh)  # (1,1,H,W) bool

    keep_np = keep[0, 0].numpy()
    hm_np   = heatmap.numpy()
    rad_np  = radmap.numpy()

    ys, xs = np.where(keep_np)
    peaks = []
    for y, x in zip(ys, xs):
        r_norm = float(rad_np[y, x])
        peaks.append((int(x), int(y), r_norm))
    return peaks


def decode_radius_km(r_norm: float) -> float:
    """Convert normalised radius [0,1] → km."""
    return r_norm * (RADIUS_MAX_KM - RADIUS_MIN_KM) + RADIUS_MIN_KM


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def match_craters(
    preds: List[Tuple[int, int, float]],
    gts:   List[Tuple[int, int, float]],   # (cx_out, cy_out, r_norm)
    max_dist_out: float = 4.0,             # max centre distance in output pixels
) -> Tuple[int, int, int, List[float]]:
    """
    Greedy matching by centre distance (nearest-first).

    Returns (TP, FP, FN, list_of_|r_pred_km - r_gt_km|_for_TP_pairs)
    """
    matched_gt  = set()
    matched_pred = set()
    radius_errors = []

    # Build distance matrix
    if len(preds) == 0 or len(gts) == 0:
        return 0, len(preds), len(gts), []

    # Sort pairs by distance (greedy closest-first)
    pairs = []
    for i, (px, py, pr) in enumerate(preds):
        for j, (gx, gy, gr) in enumerate(gts):
            dist = np.hypot(px - gx, py - gy)
            if dist <= max_dist_out:
                pairs.append((dist, i, j))
    pairs.sort()

    for dist, i, j in pairs:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        r_pred_km = decode_radius_km(preds[i][2])
        r_gt_km   = decode_radius_km(gts[j][2])
        radius_errors.append(abs(r_pred_km - r_gt_km))

    tp = len(matched_pred)
    fp = len(preds)  - tp
    fn = len(gts)    - tp
    return tp, fp, fn, radius_errors


# ---------------------------------------------------------------------------
# Batch metric computation
# ---------------------------------------------------------------------------
def compute_metrics(
    pred_hm:   torch.Tensor,   # (B, 1, H, W)
    pred_rad:  torch.Tensor,   # (B, 1, H, W)
    gt_hm:     torch.Tensor,   # (B, 1, H, W)
    gt_rad:    torch.Tensor,   # (B, 1, H, W)
    regmask:   torch.Tensor,   # (B, 1, H, W)
    hm_thresh: float = 0.3,
    nms_pool:  int   = 5,
    max_dist_out: float = 4.0,
) -> dict:
    """
    Compute detection + radius metrics over a batch.

    Returns dict:
      precision, recall, f1     — detection quality
      radius_mae_km             — mean |r_pred - r_gt| in km (TP pairs only)
      tp, fp, fn                — aggregate counts
    """
    pred_hm  = pred_hm.detach().cpu()
    pred_rad = pred_rad.detach().cpu()
    gt_hm    = gt_hm.detach().cpu()
    gt_rad   = gt_rad.detach().cpu()
    regmask  = regmask.detach().cpu()

    B = pred_hm.shape[0]
    total_tp, total_fp, total_fn = 0, 0, 0
    all_rad_errors: List[float] = []

    for b in range(B):
        # Predicted peaks
        pred_peaks = extract_peaks(pred_hm[b, 0], pred_rad[b, 0],
                                   hm_thresh=hm_thresh, nms_pool=nms_pool)

        # Ground-truth centres from regmask
        mask_np = regmask[b, 0].numpy()
        grad_np = gt_rad[b, 0].numpy()
        ys, xs  = np.where(mask_np > 0.5)
        gt_peaks = [(int(x), int(y), float(grad_np[y, x])) for y, x in zip(ys, xs)]

        tp, fp, fn, rad_err = match_craters(pred_peaks, gt_peaks, max_dist_out)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_rad_errors.extend(rad_err)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = (2 * precision * recall / max(precision + recall, 1e-6))
    radius_mae = float(np.mean(all_rad_errors)) if all_rad_errors else 0.0

    return {
        "precision":     precision,
        "recall":        recall,
        "f1":            f1,
        "radius_mae_km": radius_mae,
        "tp":            total_tp,
        "fp":            total_fp,
        "fn":            total_fn,
    }
