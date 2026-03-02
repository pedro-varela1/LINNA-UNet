"""
ShuffleNetV2-based crater detector.

Backbone  : ShuffleNetV2 x1.0  (1-channel grayscale, pretrained RGB averaged)
Head      : GAP → FC(256, ReLU) → two parallel branches
              pred_head  → (N_MAX, 3)  sigmoid  [cx, cy, r]  all normalised [0,1]
              conf_head  → (N_MAX,)    raw logits

~2.3 M parameters  (~12× fewer than Faster R-CNN version)
Inference: ~4 ms per image on GPU  (vs ~30-60 ms for Faster R-CNN)
"""

import torch, torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

from dataload import N_MAX

BACKBONE_OUT = 1024   # ShuffleNetV2 x1.0 output channels after conv5


class CraterShuffleNet(nn.Module):
    def __init__(self, pretrained=True, n_max=N_MAX):
        super().__init__()
        self.n_max = n_max

        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        backbone = shufflenet_v2_x1_0(weights=weights)

        # Patch conv1: 3-ch → 1-ch  (average RGB pretrained weights)
        old = backbone.conv1[0]
        new_conv = nn.Conv2d(1, old.out_channels,
                             kernel_size=old.kernel_size, stride=old.stride,
                             padding=old.padding, bias=False)
        if pretrained:
            new_conv.weight.data = old.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1[0] = new_conv

        # Feature extractor: everything except the final FC classifier
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.maxpool,
            backbone.stage2,
            backbone.stage3,
            backbone.stage4,
            backbone.conv5,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Shared neck
        self.neck = nn.Sequential(
            nn.Linear(BACKBONE_OUT, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Prediction head: cx, cy, r — all sigmoid → [0, 1]
        self.pred_head = nn.Sequential(
            nn.Linear(256, n_max * 3),
            nn.Sigmoid(),
        )

        # Confidence head: raw logits, one per slot
        self.conf_head = nn.Linear(256, n_max)

    def forward(self, x):
        """
        x : (B, 1, H, W)
        Returns:
            preds  (B, N_MAX, 3)   sigmoid  [cx, cy, r]
            logits (B, N_MAX)      raw confidence logits
        """
        feat   = self.pool(self.features(x)).flatten(1)   # (B, 1024)
        neck   = self.neck(feat)                           # (B, 256)
        preds  = self.pred_head(neck).view(-1, self.n_max, 3)
        logits = self.conf_head(neck)
        return preds, logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = CraterShuffleNet(pretrained=False)
    print(f"Parameters: {model.count_parameters():,}")
    x = torch.randn(2, 1, 512, 512)
    preds, logits = model(x)
    print(f"preds  {preds.shape}   [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"logits {logits.shape}")
