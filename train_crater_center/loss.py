"""
Loss function for CraterShuffleNet.

  L = w_conf * BCE(logits, mask)  +  w_reg * SmoothL1(pred[valid], gt[valid])

BCE is computed over all N_MAX slots (1 = crater present, 0 = empty).
SmoothL1 regression is computed only on valid (mask=True) slots.
"""

import torch, torch.nn as nn

_bce = nn.BCEWithLogitsLoss()
_sl1 = nn.SmoothL1Loss(reduction="mean")


class CraterLoss(nn.Module):
    def __init__(self, w_reg=2.0, w_conf=1.0):
        super().__init__()
        self.w_reg  = w_reg
        self.w_conf = w_conf

    def forward(self, preds, logits, gt_boxes, mask):
        """
        preds   : (B, N_MAX, 3)  sigmoid  [cx, cy, r]
        logits  : (B, N_MAX)     raw logits
        gt_boxes: (B, N_MAX, 3)  ground-truth [cx, cy, r]
        mask    : (B, N_MAX)     bool  — True = valid slot
        """
        # Confidence loss over all slots
        l_conf = _bce(logits, mask.float())

        # Regression loss only on valid slots
        if mask.any():
            l_reg = _sl1(preds[mask], gt_boxes[mask])
        else:
            l_reg = preds.sum() * 0.0   # gradient flows but value is 0

        total = self.w_conf * l_conf + self.w_reg * l_reg
        return total, l_conf.detach(), l_reg.detach()


if __name__ == "__main__":
    B, N = 4, 16
    preds   = torch.sigmoid(torch.randn(B, N, 3))
    logits  = torch.randn(B, N)
    gt      = torch.rand(B, N, 3)
    mask    = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :3] = True
    loss_fn = CraterLoss()
    total, lc, lr = loss_fn(preds, logits, gt, mask)
    print(f"total={total:.4f}  conf={lc:.4f}  reg={lr:.4f}")
