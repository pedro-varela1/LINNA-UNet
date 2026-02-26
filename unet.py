import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResNetUp(nn.Module):
    """
    Bloco de Upscaling adaptado para as dimensões da ResNet.
    Aceita canais de entrada (vibr do decoder) e canais de skip (vindo do encoder).
    """
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        
        self.bilinear = bilinear

        # Upsample
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Nota: Se usar transpose, precisaria ajustar os canais de entrada da conv abaixo

        # Após concatenar (in_channels + skip_channels), reduzimos para out_channels
        # Nota: O in_channels aqui se refere ao tamanho DEPOIS do upsample (que não muda canais no modo bilinear)
        
        # Total de canais entrando na conv: in_channels (do upsample) + skip_channels (da resnet)
        total_in_channels = in_channels + skip_channels
        
        self.conv = DoubleConv(total_in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Input vindo de baixo (Decoder)
        x2: Skip connection vindo da ResNet (Encoder)
        """
        x1 = self.up(x1)
        
        # Tratamento de Padding para dimensões ímpares ou erros de arredondamento
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenação: x2 (skip) + x1 (upsampled)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LunarUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=6, bilinear=True):
        super(LunarUNet, self).__init__()
        self.n_channels = n_channels
        # Saída: 6 valores por frame
        #   [xc, yc, zc]          => coordenadas do centro na superfície (tanh -> [-1,1])
        #   [width, height, alt]  => dimensões e altitude (sigmoid -> [0,1])
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- 1. Encoder: ResNet18 Pré-treinada (mais leve que ResNet50) ---
        try:
            weights = models.ResNet18_Weights.DEFAULT
            self.resnet = models.resnet18(weights=weights)
        except:
            self.resnet = models.resnet18(pretrained=True)

        # Adaptação para Grayscale (1 canal) mantendo os pesos
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.resnet.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        # Camadas do Encoder (ResNet18)
        # Stem    -> 64 ch  (enc0)
        # layer1  -> 64 ch  (enc2)   ← BasicBlock, sem expansão
        # layer2  -> 128 ch (enc3)
        # layer3  -> 256 ch (enc4)
        # layer4  -> 512 ch (enc5)   ← Bottleneck
        self.enc0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.enc1 = self.resnet.maxpool
        self.enc2 = self.resnet.layer1   # 64 ch
        self.enc3 = self.resnet.layer2   # 128 ch
        self.enc4 = self.resnet.layer3   # 256 ch
        self.enc5 = self.resnet.layer4   # 512 ch

        # --- 2. Decoder ---
        # Up1: 512 + skip 256  -> 256
        self.up1 = ResNetUp(in_channels=512,  skip_channels=256, out_channels=256, bilinear=bilinear)
        # Up2: 256 + skip 128  -> 128
        self.up2 = ResNetUp(in_channels=256,  skip_channels=128, out_channels=128, bilinear=bilinear)
        # Up3: 128 + skip 64   -> 64
        self.up3 = ResNetUp(in_channels=128,  skip_channels=64,  out_channels=64,  bilinear=bilinear)
        # Up4: 64  + skip 64   -> 32  (skip = Stem)
        self.up4 = ResNetUp(in_channels=64,   skip_channels=64,  out_channels=32,  bilinear=bilinear)

        # Upsample final (restaura resolução 2× cortada pelo stem)
        self.up5_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up5_conv = DoubleConv(32, 16)

        # Global Average Pooling + FC
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        # --- Encoder Path ---
        # x: (B, 1, H, W)
        x0 = self.enc0(x)      # Stem:   (B, 64, H/2,  W/2)  -> SKIP 1
        x1 = self.enc1(x0)     # Pool:   (B, 64, H/4,  W/4)
        x2 = self.enc2(x1)     # layer1: (B, 64, H/4,  W/4)  -> SKIP 2
        x3 = self.enc3(x2)     # layer2: (B,128, H/8,  W/8)  -> SKIP 3
        x4 = self.enc4(x3)     # layer3: (B,256, H/16, W/16) -> SKIP 4
        x5 = self.enc5(x4)     # layer4: (B,512, H/32, W/32) -> Bottleneck

        # --- Decoder Path ---
        x = self.up1(x5, x4)   # 512+256  -> 256
        x = self.up2(x,  x3)   # 256+128  -> 128
        x = self.up3(x,  x2)   # 128+64   -> 64
        x = self.up4(x,  x0)   # 64+64    -> 32  (skip = Stem)

        # Upsample final
        x = self.up5_upsample(x)
        x = self.up5_conv(x)   # -> 16 ch

        # Global Average Pooling + FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)   # (B, 16)
        x = self.fc(x)               # (B, n_classes)

        # Ativações diferenciadas por grupo de saída:
        #   [:3]  - XYZ do centro da imagem na superfície: tanh  -> [-1, 1]
        #   [3:]  - largura, altura (km) e altitude (km): sigmoid -> [ 0, 1]
        xyz_out = torch.tanh(x[:, :3])
        dim_out = torch.sigmoid(x[:, 3:])
        return torch.cat([xyz_out, dim_out], dim=1)

# --- Bloco de Teste Rápido ---
if __name__ == "__main__":
    # Teste de integridade das dimensões
    print("🔄 Inicializando LunarUNet com ResNet50 Backbone...")
    
    # 1. Criar dummy input (Grayscale 512x512)
    dummy_input = torch.randn(2, 1, 512, 512)
    
    # 2. Instanciar modelo
    model = LunarUNet(n_channels=1, n_classes=6)

    # 3. Forward Pass
    try:
        output = model(dummy_input)
        print(f"✅ Sucesso!")
        print(f"   Input shape:  {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")

        assert output.shape == (2, 6), "Erro: Dimensão de saída incorreta!"
        print(f"   XYZ range (tanh) :  [{output[:, :3].min().item():.3f}, {output[:, :3].max().item():.3f}]")
        print(f"   W/H/Alt range (sigmoid): [{output[:, 3:].min().item():.3f}, {output[:, 3:].max().item():.3f}]")
    except Exception as e:
        print(f"❌ Erro durante o forward pass: {e}")