"""
Evaluation metrics for altitude-only prediction.

Expected tensor format:
    preds   (B,) — z-score normalized altitude  (model output)
    targets (B,) — z-score normalized altitude  (ground truth)

Pass alt_mean and alt_std (km) to recover physical values.
"""

import torch


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    alt_mean: float,
    alt_std:  float,
    threshold_km: float = 5.0,
) -> dict:
    """
    Compute altitude prediction metrics for a batch.

    Args:
        preds        : Z-score model predictions  (B,).
        targets      : Z-score ground truth       (B,).
        alt_mean     : Training-set mean altitude (km) used for z-score.
        alt_std      : Training-set std  altitude (km) used for z-score.
        threshold_km : Error threshold (km) used to compute accuracy.

    Returns:
        dict with:
            mae_km          — Mean Absolute Error in km.
            rmse_km         — Root Mean Squared Error in km.
            accuracy_percent— % of samples with |error| < threshold_km.
            threshold_used_km
    """
    pred_km = preds   * alt_std + alt_mean
    targ_km = targets * alt_std + alt_mean

    abs_error = torch.abs(pred_km - targ_km)

    mae_km   = abs_error.mean().item()
    rmse_km  = (abs_error**2).mean().sqrt().item()
    accuracy = (abs_error < threshold_km).float().mean().item() * 100.0

    return {
        "mae_km":           mae_km,
        "rmse_km":          rmse_km,
        "accuracy_percent": accuracy,
        "threshold_used_km": threshold_km,
    }


# --- Quick sanity check ---
if __name__ == "__main__":
    print("🧪  Testing compute_metrics...")

    # Simulate z-score labels: mean=60 km, std=30 km, true altitude=50 km
    ALT_MEAN, ALT_STD = 60.0, 30.0
    targets = torch.full((8,), (50.0 - ALT_MEAN) / ALT_STD)
    preds   = targets + torch.randn(8) * (3.0 / ALT_STD)  # ±3 km noise

    m = compute_metrics(preds, targets, alt_mean=ALT_MEAN, alt_std=ALT_STD, threshold_km=5.0)

    print(f"   MAE       : {m['mae_km']:.2f} km")
    print(f"   RMSE      : {m['rmse_km']:.2f} km")
    print(f"   Accuracy  : {m['accuracy_percent']:.1f}%  (threshold={m['threshold_used_km']} km)")
