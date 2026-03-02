"""
MobileNetV2-based altitude regressor for LINNA.

Input : grayscale image  (B, 1, H, W)
Output: z-score normalized altitude — denormalize with (pred * alt_std + alt_mean) to get km.
"""

import torch.nn as nn
from torchvision.models import mobilenet_v2


class AltitudeMobileNet(nn.Module):
    """
    MobileNetV2 backbone adapted for single-channel input and scalar altitude regression.

    Modifications from the vanilla MobileNetV2:
        - features[0][0]: 3 channels → 1 channel (grayscale input).
        - classifier    : 1000 classes → [1280 → 64 → 1], unbounded output for z-score.
    """

    def __init__(self):
        super().__init__()

        backbone = mobilenet_v2(weights=None)

        # Adapt first conv to accept single-channel (grayscale) images
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Replace classifier head with a lightweight altitude regressor
        # No Sigmoid: z-score labels are unbounded (can be negative or > 1)
        backbone.classifier = nn.Sequential(
            nn.Linear(1280, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.net = backbone

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B, 1) → (B,)


# --- Quick sanity check ---
if __name__ == "__main__":
    import torch

    model = AltitudeMobileNet()
    dummy = torch.randn(4, 1, 224, 224)
    out = model(dummy)

    print("✅  Model OK")
    print(f"   Input  shape : {dummy.shape}")
    print(f"   Output shape : {out.shape}")   # (4,)
    print(f"   Output range : [{out.min():.3f}, {out.max():.3f}]  (should be in [0, 1])")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters   : {total_params:,}")
