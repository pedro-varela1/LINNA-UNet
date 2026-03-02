"""
CenterNet-style crater detector built on a ResNet-18 backbone.

Architecture
------------
  Encoder : ResNet-18 (grayscale input, 1 channel)
            Produces feature maps at strides 4, 8, 16, 32.
  Decoder : Three transposed-conv up-sampling blocks (32→16→8→4)
            with residual addition of encoder skip features (FPN-like).
  Heads   : Two independent 3×3 + 1×1 convolutional heads at stride-4:
              - heatmap_head → σ(·)   shape (B, 1, OUT_H, OUT_W)
              - radius_head  → σ(·)   shape (B, 1, OUT_H, OUT_W)

Output resolution is IMG_SIZE / 4 = 128 × 128 (same as dataload OUT_SIZE).

Usage
-----
    from model import CraterCenterNet
    model = CraterCenterNet(pretrained=True)    # pretrained ResNet-18 weights
    heatmap, radmap = model(images)             # images: (B, 1, 512, 512)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------
class _UpBlock(nn.Module):
    """Transposed-conv (×2) + BN + ReLU, then optional skip-feature fusion."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # 1×1 projection for skip features (if skip_ch != out_ch)
        self.skip_proj = (
            nn.Conv2d(skip_ch, out_ch, 1, bias=False) if skip_ch != out_ch else nn.Identity()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        skip = self.skip_proj(skip)
        x    = x + skip          # residual add (FPN-style)
        return self.fuse(x)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
class _Head(nn.Module):
    """3×3 conv → ReLU → 1×1 conv → sigmoid."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class CraterCenterNet(nn.Module):
    """
    ResNet-18 backbone + FPN-like decoder + two output heads.

    Inputs
      images : (B, 1, H, W)  grayscale lunar images

    Outputs
      heatmap : (B, 1, H/4, W/4)   crater-centre probability    [0, 1]
      radmap  : (B, 1, H/4, W/4)   normalised radius            [0, 1]
                    decoded km = out * (RADIUS_MAX - RADIUS_MIN) + RADIUS_MIN
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # --- Backbone ---
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Replace first conv: 3-channel RGB → 1-channel grayscale.
        # Average pre-trained weights across the channel dimension so we
        # do not throw away ImageNet knowledge entirely.
        old_w = backbone.conv1.weight.data        # (64, 3, 7, 7)
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = old_w.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # Expose encoder stages (stride × from input):
        self.enc_stem  = nn.Sequential(backbone.conv1, backbone.bn1,
                                       backbone.relu, backbone.maxpool)   # s4,  64ch
        self.enc_layer1 = backbone.layer1   # s4,   64ch
        self.enc_layer2 = backbone.layer2   # s8,  128ch
        self.enc_layer3 = backbone.layer3   # s16, 256ch
        self.enc_layer4 = backbone.layer4   # s32, 512ch

        # --- Decoder (s32→s16→s8→s4) ---
        FEAT_CH = 256  # uniform decoder channel width
        self.up4 = _UpBlock(512, 256, FEAT_CH)   # s32→s16, skip from layer3 (256ch)
        self.up3 = _UpBlock(FEAT_CH, 128, FEAT_CH)  # s16→s8,  skip from layer2 (128ch)
        self.up2 = _UpBlock(FEAT_CH,  64, FEAT_CH)  # s8→s4,   skip from layer1 (64ch)

        # --- Output heads ---
        self.heatmap_head = _Head(FEAT_CH)  # bias set after _init_weights
        self.radius_head  = _Head(FEAT_CH)

        # Initialise decoder & head weights (generic kaiming init)
        self._init_weights()

        # Override heatmap-head final bias: -2.19 → sigmoid ≈ 0.10
        # (most pixels are background, so start with low confidence)
        nn.init.constant_(self.heatmap_head.net[-2].bias, -2.19)

    def _init_weights(self):
        for m in [self.up4, self.up3, self.up2, self.heatmap_head, self.radius_head]:
            for layer in m.modules():
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        # Encoder
        s = self.enc_stem(x)       # (B,  64, H/4,  W/4)
        e1 = self.enc_layer1(s)    # (B,  64, H/4,  W/4)
        e2 = self.enc_layer2(e1)   # (B, 128, H/8,  W/8)
        e3 = self.enc_layer3(e2)   # (B, 256, H/16, W/16)
        e4 = self.enc_layer4(e3)   # (B, 512, H/32, W/32)

        # Decoder
        d = self.up4(e4, e3)      # (B, 256, H/16, W/16)
        d = self.up3(d,  e2)      # (B, 256, H/8,  W/8)
        d = self.up2(d,  e1)      # (B, 256, H/4,  W/4)

        # Heads
        heatmap = self.heatmap_head(d)   # (B, 1, H/4, W/4)
        radmap  = self.radius_head(d)    # (B, 1, H/4, W/4)

        return heatmap, radmap


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = CraterCenterNet(pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params   : {total:,}")
    print(f"Trainable      : {trainable:,}")

    x = torch.randn(2, 1, 512, 512)
    hm, rm = model(x)
    print(f"Input  : {x.shape}")
    print(f"Heatmap: {hm.shape}  [{hm.min():.3f}, {hm.max():.3f}]")
    print(f"Radmap : {rm.shape}  [{rm.min():.3f}, {rm.max():.3f}]")
