"""
Evaluation metrics for altitude-only prediction.

Expected tensor format:
    preds   (B,) — normalized altitude in [0, 1]  (model output)
    targets (B,) — normalized altitude in [0, 1]  (ground truth)

Multiply by ALT_MAX_KM (120 km) to recover physical values.
"""

import torch

ALT_MAX_KM = 120.0


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold_km: float = 5.0) -> dict:
    """
    Compute altitude prediction metrics for a batch.

    Args:
        preds        : Normalized model predictions  (B,) in [0, 1].
        targets      : Normalized ground truth       (B,) in [0, 1].
        threshold_km : Error threshold (km) used to compute accuracy.

    Returns:
        dict with:
            mae_km          — Mean Absolute Error in km.
            rmse_km         — Root Mean Squared Error in km.
            accuracy_percent— % of samples with |error| < threshold_km.
            threshold_used_km
    """
    pred_km = preds   * ALT_MAX_KM
    targ_km = targets * ALT_MAX_KM

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

    # Simulate a batch: ground truth at 50 km, predictions slightly off
    targets = torch.full((8,), 50.0 / ALT_MAX_KM)
    preds   = targets + torch.randn(8) * (3.0 / ALT_MAX_KM)  # ±3 km noise

    m = compute_metrics(preds, targets, threshold_km=5.0)

    print(f"   MAE       : {m['mae_km']:.2f} km")
    print(f"   RMSE      : {m['rmse_km']:.2f} km")
    print(f"   Accuracy  : {m['accuracy_percent']:.1f}%  (threshold={m['threshold_used_km']} km)")
