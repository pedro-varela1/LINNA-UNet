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
    def __init__(self, n_channels=1, n_classes=12, bilinear=True):
        super(LunarUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes  # Agora são 12: 4 pontos × 3 coordenadas
        self.bilinear = bilinear

        # --- 1. Encoder: ResNet50 Pré-treinada ---
        # Usamos weights='DEFAULT' para pegar os melhores pesos disponíveis
        try:
            weights = models.ResNet50_Weights.DEFAULT
            self.resnet = models.resnet50(weights=weights)
        except:
            # Fallback para versões mais antigas do torch
            self.resnet = models.resnet50(pretrained=True)

        # Adaptação para Grayscale (1 canal) mantendo os pesos
        # A conv1 original é (64, 3, 7, 7). Vamos somar os pesos dos 3 canais para virar (64, 1, 7, 7)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            # Soma os pesos RGB para criar um filtro de intensidade robusto
            self.resnet.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)
        
        # Camadas do Encoder (Extraídas da ResNet)
        self.enc0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu) # Stem
        self.enc1 = self.resnet.maxpool # Pool
        self.enc2 = self.resnet.layer1  # 256 canais
        self.enc3 = self.resnet.layer2  # 512 canais
        self.enc4 = self.resnet.layer3  # 1024 canais
        self.enc5 = self.resnet.layer4  # 2048 canais (Bottleneck)

        # --- 2. Decoder ---
        # Canais da ResNet50:
        # Layer 4 (Bottleneck): 2048
        # Layer 3 (Skip 4): 1024
        # Layer 2 (Skip 3): 512
        # Layer 1 (Skip 2): 256
        # Stem    (Skip 1): 64  (Saída do enc0)
        
        # Up1: Input 2048 + Skip 1024 -> Out 1024
        self.up1 = ResNetUp(in_channels=2048, skip_channels=1024, out_channels=1024, bilinear=bilinear)
        
        # Up2: Input 1024 + Skip 512 -> Out 512
        self.up2 = ResNetUp(in_channels=1024, skip_channels=512,  out_channels=512,  bilinear=bilinear)
        
        # Up3: Input 512  + Skip 256 -> Out 256
        self.up3 = ResNetUp(in_channels=512,  skip_channels=256,  out_channels=256,  bilinear=bilinear)
        
        # Up4: Input 256  + Skip 64  -> Out 64
        self.up4 = ResNetUp(in_channels=256,  skip_channels=64,   out_channels=64,   bilinear=bilinear)
        
        # Up5: Input 64 (Sem skip, apenas upsample final para restaurar resolução total)
        # ResNet reduz a imagem em 2x logo no início. Precisamos de um bloco final.
        self.up5_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up5_conv = DoubleConv(64, 32) # Reduz para 32 canais antes da saída

        # Global Average Pooling + FC para produzir 12 valores (4 pontos × 3 coordenadas)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        # --- Encoder Path ---
        # x: (B, 1, 512, 512)
        
        x0 = self.enc0(x)      # Stem: (B, 64, 256, 256) -> SKIP 1
        x1 = self.enc1(x0)     # Maxpool: (B, 64, 128, 128)
        
        x2 = self.enc2(x1)     # Layer1: (B, 256, 128, 128) -> SKIP 2
        x3 = self.enc3(x2)     # Layer2: (B, 512, 64, 64)   -> SKIP 3
        x4 = self.enc4(x3)     # Layer3: (B, 1024, 32, 32)  -> SKIP 4
        x5 = self.enc5(x4)     # Layer4: (B, 2048, 16, 16)  -> Bottleneck

        # --- Decoder Path ---
        x = self.up1(x5, x4)   # 16->32. Cat x4. Out: 1024
        x = self.up2(x, x3)    # 32->64. Cat x3. Out: 512
        x = self.up3(x, x2)    # 64->128. Cat x2. Out: 256
        
        # Atenção: x0 é o skip da Stem (antes do maxpool)
        x = self.up4(x, x0)    # 128->256. Cat x0. Out: 64

        # Upsample Final (256->512)
        x = self.up5_upsample(x)
        x = self.up5_conv(x)
        
        # Global Average Pooling: (B, 32, 512, 512) -> (B, 32, 1, 1)
        x = self.global_pool(x)
        # Flatten: (B, 32, 1, 1) -> (B, 32)
        x = x.view(x.size(0), -1)
        
        # FC: (B, 32) -> (B, 12)
        x = self.fc(x)
        
        # Ativação Tanh para garantir output em [-1, 1]
        # Isso corresponde ao range das coordenadas cartesianas normalizadas
        output = torch.tanh(x)
        
        return output

# --- Bloco de Teste Rápido ---
if __name__ == "__main__":
    # Teste de integridade das dimensões
    print("🔄 Inicializando LunarUNet com ResNet50 Backbone...")
    
    # 1. Criar dummy input (Grayscale 512x512)
    dummy_input = torch.randn(2, 1, 512, 512)
    
    # 2. Instanciar modelo
    model = LunarUNet(n_channels=1, n_classes=12)
    
    # 3. Forward Pass
    try:
        output = model(dummy_input)
        print(f"✅ Sucesso!")
        print(f"   Input shape:  {dummy_input.shape}")
        print(f"   Output shape: {output.shape}") 
        
        # Verificação extra
        assert output.shape == (2, 12), "Erro: Dimensão de saída incorreta!"
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    except Exception as e:
        print(f"❌ Erro durante o forward pass: {e}")