"""
ResNet18-based altitude regressor for lunar landing.

Input : grayscale image  (B, 1, H, W)
Output: normalized altitude in [0, 1]  — multiply by ALT_MAX_KM (120 km) to get km.
"""

import torch.nn as nn
from torchvision.models import resnet18


class AltitudeResNet(nn.Module):
    """
    ResNet18 backbone adapted for single-channel input and scalar altitude regression.

    Modifications from the vanilla ResNet18:
        - conv1: 3 channels → 1 channel (grayscale input).
        - fc   : 1000 classes → [512 → 64 → 1] with Sigmoid output in [0, 1].
    """

    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=None)

        # Adapt first conv to accept single-channel (grayscale) images
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace classifier head with a lightweight altitude regressor
        backbone.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Clamps output to [0, 1]
        )

        self.net = backbone

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B, 1) → (B,)


# --- Quick sanity check ---
if __name__ == "__main__":
    import torch

    model = AltitudeResNet()
    dummy = torch.randn(4, 1, 224, 224)
    out = model(dummy)

    print("✅  Model OK")
    print(f"   Input  shape : {dummy.shape}")
    print(f"   Output shape : {out.shape}")   # (4,)
    print(f"   Output range : [{out.min():.3f}, {out.max():.3f}]  (should be in [0, 1])")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters   : {total_params:,}")
