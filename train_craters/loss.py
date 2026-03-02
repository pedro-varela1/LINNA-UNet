"""
Loss functions for CenterNet-style crater detection.

  CraterLoss
  ----------
  Combines two terms:

  1. Heatmap loss  — Modified Focal Loss (CornerNet / CenterNet formulation).
     Handles extreme foreground/background imbalance by down-weighting easy
     negatives *and* near-positive pixels (those inside the Gaussian spread):

       L_hm = -Σ  (1-p)^α · log(p)          if y == 1  (crater peak)
                  (1-y)^β · p^α · log(1-p)   otherwise  (y is Gaussian label)

     α = 2, β = 4  (original CenterNet defaults).

  2. Radius loss  — Smooth-L1 (Huber) between predicted and ground-truth
     normalised radius, evaluated *only* at the exact crater-centre pixels
     (regmask == 1).  If there are no craters in the batch the loss is 0.

  Total loss = w_hm * L_hm + w_rad * L_rad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedFocalLoss(nn.Module):
    """
    Focal loss for dense heatmap regression (CenterNet / CornerNet variant).

    Args:
        alpha : exponent on (1-p) and (1-pred) — controls hard-example focus.
        beta  : exponent on (1-y)              — down-weights near-Gaussian pix.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B, 1, H, W) — sigmoid output of heatmap head [0, 1]
            target : (B, 1, H, W) — Gaussian heatmap labels [0, 1]
        """
        eps = 1e-6
        pred   = pred.clamp(eps, 1.0 - eps)
        target = target.clamp(0.0, 1.0)

        # Positive pixels: where target == 1  (exact Gaussian peak)
        pos_mask = (target == 1.0).float()
        neg_mask = 1.0 - pos_mask

        pos_loss = (
            pos_mask
            * torch.pow(1.0 - pred, self.alpha)
            * torch.log(pred)
        )

        neg_loss = (
            neg_mask
            * torch.pow(1.0 - target, self.beta)
            * torch.pow(pred, self.alpha)
            * torch.log(1.0 - pred)
        )

        num_pos = pos_mask.sum().clamp(min=1.0)
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class CraterLoss(nn.Module):
    """
    Combined loss for CraterCenterNet.

    Args:
        w_hm  : weight for the heatmap focal loss  (default 1.0)
        w_rad : weight for the radius smooth-L1    (default 1.0)
    """

    def __init__(self, w_hm: float = 1.0, w_rad: float = 1.0):
        super().__init__()
        self.w_hm  = w_hm
        self.w_rad = w_rad
        self.focal = ModifiedFocalLoss(alpha=2.0, beta=4.0)

    def forward(
        self,
        pred_hm:  torch.Tensor,   # (B, 1, H, W)  predicted heatmap
        pred_rad: torch.Tensor,   # (B, 1, H, W)  predicted radius map
        gt_hm:    torch.Tensor,   # (B, 1, H, W)  target heatmap (Gaussian labels)
        gt_rad:   torch.Tensor,   # (B, 1, H, W)  target radius  (norm, only at peaks)
        regmask:  torch.Tensor,   # (B, 1, H, W)  1 at crater centres, 0 elsewhere
    ):
        # --- Heatmap focal loss ---
        l_hm = self.focal(pred_hm, gt_hm)

        # --- Radius smooth-L1 only at crater centres ---
        num_craters = regmask.sum().clamp(min=1.0)
        l_rad = (
            F.smooth_l1_loss(pred_rad * regmask, gt_rad * regmask, reduction="sum")
            / num_craters
        )

        total = self.w_hm * l_hm + self.w_rad * l_rad
        return total, l_hm.detach(), l_rad.detach()
