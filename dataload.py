import os
import glob
import random
import cv2
import torch
import numpy as np
import OpenEXR
import Imath
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  # mantido para outros usos futuros
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constante do raio da Lua em km
MOON_RADIUS_KM = 1737.4

# Range de altitude do satélite acima da superfície (km)
# Altitude 0 km => sobre a superfície (r = 1737.4 km)
# Altitude MAX km => altitude máxima esperada
ALT_MIN_KM  = 0.0
ALT_MAX_KM  = 120.0   # Ajuste se o dataset tiver altitudes maiores

def latlon_to_cartesian(lat, lon, alt):
    """
    Converte coordenadas geográficas (lat, lon, altitude) para coordenadas cartesianas 3D (x, y, z).
    
    Args:
        lat: Latitude em graus [-90, 90]
        lon: Longitude em graus [0, 360]
        alt: Altitude em km acima da superfície
        
    Returns:
        x, y, z: Coordenadas cartesianas em km
    """
    # Raio total (raio da lua + altitude)
    r = MOON_RADIUS_KM + alt
    
    # Converter para radianos
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Conversão para coordenadas cartesianas
    # Sistema: x aponta para lon=0, y aponta para lon=90, z aponta para polo norte
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    
    return x, y, z

def cartesian_to_latlon(x, y, z):
    """
    Converte coordenadas cartesianas 3D (x, y, z) de volta para (lat, lon, altitude).
    
    Args:
        x, y, z: Coordenadas cartesianas em km
        
    Returns:
        lat, lon, alt: Latitude (graus), Longitude (graus), Altitude (km)
    """
    # Calcular raio total
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Altitude = raio total - raio da lua
    alt = r - MOON_RADIUS_KM
    
    # Latitude (em radianos, depois converter para graus)
    lat_rad = np.arcsin(np.clip(z / r, -1.0, 1.0))
    lat = np.rad2deg(lat_rad)
    
    # Longitude (em radianos, depois converter para graus)
    lon_rad = np.arctan2(y, x)
    lon = np.rad2deg(lon_rad)
    
    # Garantir longitude no range [0, 360]
    lon = np.where(lon < 0, lon + 360, lon)
    
    return lat, lon, alt

class LunarDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train',
                 group_size=12, val_per_group=2, random_seed=42, max_dim_km=None):
        """
        Args:
            root_dir (str): Caminho raiz do dataset.
            transform (callable, optional): Transformações de data augmentation.
            mode (str): 'train' ou 'val'.
            group_size (int): Número de imagens por grupo/região (padrão 12).
            val_per_group (int): Quantas imagens de cada grupo vão para validação.
                São escolhidas ALEATORIAMENTE dentro do grupo (padrão 2).
            random_seed (int): Semente para reprodutibilidade do sorteio (padrão 42).
            max_dim_km (float, optional): Valor máximo de largura/altura em km para
                normalização. Se None, é calculado varrendo todos os arquivos EXR.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Criar augmentations se em modo de treino
        # Apenas augmentations FOTOMÉTRICAS — não alteram a geometria da cena
        # e portanto não requerem atualização dos targets.
        # ReplayCompose registra quais transforms foram efetivamente aplicados.
        if mode == 'train':
            self.augmentations = A.ReplayCompose([
                # Variação de brilho/contraste (simula ângulo solar diferente)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                # Variação de gama (simula diferenças de exposição da câmera)
                A.RandomGamma(
                    gamma_limit=(80, 120), p=0.4),
                # Ruído gaussiano (ruído do sensor)
                A.GaussNoise(
                    noise_scale_factor=1.0, p=0.4),
                # Desfoque leve (movimento / foco imperfeito)
                A.GaussianBlur(
                    blur_limit=(3, 5), p=0.3),
            ])
        else:
            self.augmentations = None
        self._last_aug_replay = None  # registra replay da última chamada a __getitem__
        
        # 1. Coletar arquivos por nome
        img_files = glob.glob(os.path.join(root_dir, "img", "*.png"))
        exr_files = glob.glob(os.path.join(root_dir, "lat_lon_exr", "*.exr"))
        alt_files = glob.glob(os.path.join(root_dir, "altimeter", "*.txt"))
        
        # Criar dicionários por nome base
        img_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
        exr_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in exr_files}
        alt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in alt_files}
        
        # Encontrar nomes comuns
        common_names = sorted(set(img_dict.keys()) & set(exr_dict.keys()) & set(alt_dict.keys()))
        
        self.img_files = [img_dict[name] for name in common_names]
        self.exr_files = [exr_dict[name] for name in common_names]
        self.alt_files = [alt_dict[name] for name in common_names]

        # 2. Divisão Treino/Validação por grupos de região
        # Cada grupo de `group_size` imagens consecutivas representa uma mesma região.
        # Dentro de cada grupo completo, `val_per_group` índices são sorteados
        # aleatoriamente para validação; os restantes vão para treino.
        # Grupos incompletos (último grupo) vão todos para treino.
        n = len(self.img_files)
        train_idx, val_idx = [], []
        rng = random.Random(random_seed)  # RNG isolado, não afeta estado global

        for group_start in range(0, n, group_size):
            group = list(range(group_start, min(group_start + group_size, n)))
            if len(group) == group_size:
                # Sorteia `val_per_group` índices aleatoriamente dentro do grupo
                chosen_val = set(rng.sample(group, val_per_group))
                for idx in group:
                    (val_idx if idx in chosen_val else train_idx).append(idx)
            else:
                # Grupo incompleto (último): tudo vai para treino
                train_idx.extend(group)

        self.active_indices = train_idx if mode == 'train' else val_idx

        n_total = len(train_idx) + len(val_idx)
        print(f"[LunarDataset/{mode}] Split por grupos: "
              f"{len(train_idx)} treino / {len(val_idx)} val "
              f"(grupos de {group_size}, {val_per_group} val/grupo sorteados, seed={random_seed}) "
              f"— {n_total}/{n} usados")

        # 3. Constantes de Normalização
        #
        # XYZ do centro: o ponto central da imagem está SEMPRE na superfície
        # (r = MOON_RADIUS_KM = 1737.4 km). Normalizamos por MOON_RADIUS_KM,
        # de modo que x, y, z ∈ [-1, 1].
        self.NORM_RADIUS = MOON_RADIUS_KM  # 1737.4 km
        #
        # Altitude: valor escalar acima da superfície ∈ [0, ALT_MAX_KM].
        # Normalizado para [0, 1] dividindo por ALT_MAX_KM.
        self.ALT_NORM = ALT_MAX_KM  # 120.0 km
        #
        # Largura / Altura (footprint na superfície):
        # Calculamos o máximo do dataset para usar como escala.
        # Se max_dim_km for fornecido externamente, reutilizamos (sem re-scan).
        if max_dim_km is not None:
            self.MAX_DIM_KM = float(max_dim_km)
        else:
            print(f"[LunarDataset/{mode}] Calculando MAX_DIM_KM varrendo {len(self.exr_files)} arquivos EXR...")
            self.MAX_DIM_KM = self._compute_max_dim()
            print(f"[LunarDataset/{mode}] MAX_DIM_KM = {self.MAX_DIM_KM:.2f} km")

    def __len__(self):
        return len(self.active_indices)

    def _load_exr_corners(self, exr_path):
        """
        Carrega um arquivo EXR e retorna (lat_map, lon_map) como arrays numpy.
        """
        exr_file = OpenEXR.InputFile(exr_path)
        header   = exr_file.header()
        dw       = header['dataWindow']
        size     = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        FLOAT    = Imath.PixelType(Imath.PixelType.FLOAT)
        r_str    = exr_file.channel('R', FLOAT)
        g_str    = exr_file.channel('G', FLOAT)
        lat_map  = np.frombuffer(r_str, dtype=np.float32).reshape(size[1], size[0])
        lon_map  = np.frombuffer(g_str, dtype=np.float32).reshape(size[1], size[0])
        return lat_map, lon_map

    def _compute_max_dim(self):
        """
        Varre TODOS os arquivos EXR do dataset (não só o split ativo) e retorna
        o valor máximo de largura/altura do footprint na superfície lunar (em km).
        """
        max_dim = 0.0
        for exr_path in self.exr_files:
            lat_map, lon_map = self._load_exr_corners(exr_path)
            H, W = lat_map.shape

            # TL, TR, BL na superfície (alt = 0)
            tl = np.array(latlon_to_cartesian(lat_map[0,   0  ], lon_map[0,   0  ], 0.0))
            tr = np.array(latlon_to_cartesian(lat_map[0,   W-1], lon_map[0,   W-1], 0.0))
            bl = np.array(latlon_to_cartesian(lat_map[H-1, 0  ], lon_map[H-1, 0  ], 0.0))

            width  = float(np.linalg.norm(tr - tl))
            height = float(np.linalg.norm(bl - tl))
            max_dim = max(max_dim, width, height)

        return max_dim if max_dim > 0.0 else 1.0

    def __getitem__(self, idx):
        # Mapeia o índice do dataset para o índice real do arquivo
        real_idx = self.active_indices[idx]

        # --- A. Carregar Input (Imagem) ---
        img_path = self.img_files[real_idx]
        image    = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # (H, W)

        # --- B. Carregar Lat/Lon Map e Altitude ---
        lat_map, lon_map = self._load_exr_corners(self.exr_files[real_idx])

        alt_path = self.alt_files[real_idx]
        with open(alt_path, 'r') as f:
            altitude_scalar = float(f.read().strip())  # km acima da superfície

        # --- C. Aplicar Augmentations (se em treino) ---
        # São apenas fotométricas: não afetam lat/lon maps nem os targets.
        # Aplicamos diretamente na imagem grayscale (H, W, 1).
        if self.augmentations is not None:
            # Albumentations espera (H, W, C) para imagens com canal explícito
            augmented = self.augmentations(image=image[:, :, np.newaxis])
            image = augmented['image'][:, :, 0]  # volta para (H, W)
            self._last_aug_replay = augmented.get('replay', None)

        # Normalizar imagem [0, 255] -> [0, 1]
        image = image.astype(np.float32) / 255.0

        # --- D. Extrair os 6 valores alvo ---
        H, W = lat_map.shape

        # D2. Cantos na superfície (alt = 0) — calculados primeiro pois o centro usa eles
        tl = np.array(latlon_to_cartesian(lat_map[0,   0  ], lon_map[0,   0  ], 0.0))
        tr = np.array(latlon_to_cartesian(lat_map[0,   W-1], lon_map[0,   W-1], 0.0))
        bl = np.array(latlon_to_cartesian(lat_map[H-1, 0  ], lon_map[H-1, 0  ], 0.0))
        br = np.array(latlon_to_cartesian(lat_map[H-1, W-1], lon_map[H-1, W-1], 0.0))

        # D1. Centro: média cartesiana dos 4 cantos, reprojetada na esfera lunar.
        # Usar o pixel central do EXR é impreciso se a imagem tiver inclinação
        # (off-nadir). A média + reprojetção garante sempre r = MOON_RADIUS_KM.
        center       = (tl + tr + bl + br) / 4.0
        r_center     = np.linalg.norm(center)
        center_unit  = center / r_center                           # vetor unitário
        xc_n, yc_n, zc_n = center_unit.tolist()                   # norma = 1 exata
        # (NORM_RADIUS == MOON_RADIUS_KM, então center_unit == center / NORM_RADIUS)

        # D2 (continuação). Largura TL↔TR e Altura TL↔BL
        width_km  = float(np.linalg.norm(tr - tl))
        height_km = float(np.linalg.norm(bl - tl))

        # Normalizar para [0, 1]
        width_n  = width_km  / self.MAX_DIM_KM
        height_n = height_km / self.MAX_DIM_KM

        # D3. Altitude do satélite (km acima da superfície) → [0, 1]
        alt_n = altitude_scalar / self.ALT_NORM

        # Target: 6 valores [xc, yc, zc, width, height, altitude]
        #   xc, yc, zc ∈ [-1, 1]   (vetor unitário na esfera — norma garantidamente = 1)
        #   width, height ∈ [0, 1]  (footprint normalizado por MAX_DIM_KM)
        #   altitude ∈ [0, 1]       (altitude normalizada por ALT_MAX_KM = 120 km)
        target = np.array([xc_n, yc_n, zc_n, width_n, height_n, alt_n], dtype=np.float32)

        # Adicionar dimensão de canal à imagem -> (1, H, W)
        image = np.expand_dims(image, axis=0)

        # Converter para Tensores PyTorch
        return torch.from_numpy(image), torch.from_numpy(target)

    def denormalize(self, tensor_pred):
        """
        Converte a saída normalizada da rede de volta às unidades físicas.

        Args:
            tensor_pred: Tensor (B, 6) -
                [xc_n, yc_n, zc_n, width_n, height_n, alt_n]
        Returns:
            Tensor (B, 6) com:
                [xc(km), yc(km), zc(km), width(km), height(km), alt(km)]
        """
        pred = tensor_pred.clone()

        # XYZ do centro (× MOON_RADIUS_KM)
        pred[:, 0] = pred[:, 0] * self.NORM_RADIUS
        pred[:, 1] = pred[:, 1] * self.NORM_RADIUS
        pred[:, 2] = pred[:, 2] * self.NORM_RADIUS

        # Largura e altura (× MAX_DIM_KM)
        pred[:, 3] = pred[:, 3] * self.MAX_DIM_KM
        pred[:, 4] = pred[:, 4] * self.MAX_DIM_KM

        # Altitude (× ALT_NORM)
        pred[:, 5] = pred[:, 5] * self.ALT_NORM

        return pred

    @staticmethod
    def denormalize_static(tensor_pred, max_dim_km, alt_norm=ALT_MAX_KM):
        """
        Versão estática de denormalize para uso sem uma instância do dataset.

        Args:
            tensor_pred: Tensor (B, 6)
            max_dim_km (float): MAX_DIM_KM usado na normalização.
            alt_norm (float): Valor de normalização da altitude (default 120.0).
        Returns:
            Tensor (B, 6) em unidades físicas.
        """
        pred = tensor_pred.clone()
        pred[:, :3] = pred[:, :3] * MOON_RADIUS_KM
        pred[:, 3:5] = pred[:, 3:5] * max_dim_km
        pred[:, 5]   = pred[:, 5]   * alt_norm
        return pred
    
    @staticmethod
    def cartesian_to_geographic(x, y, z):
        """
        Converte coordenadas cartesianas (km) para geográficas.
        Args: x, y, z podem ser tensors ou numpy arrays
        Returns: lat (graus), lon (graus), alt (km)
        """
        # Converter tensor para numpy se necessário
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            y = y.cpu().numpy()
            z = z.cpu().numpy()
            
        return cartesian_to_latlon(x, y, z)

def get_dataloaders(dataset_root, batch_size=4, group_size=12, val_per_group=2,
                    random_seed=42, num_workers=2):
    """
    Helper para criar os DataLoaders de treino e validação.

    A divisão é feita por grupos de `group_size` imagens consecutivas (mesma região).
    Dentro de cada grupo, `val_per_group` imagens são sorteadas aleatoriamente
    para validação (semente `random_seed`); as demais vão para treino.

    O MAX_DIM_KM é calculado uma única vez no dataset de treino (varrendo todos
    os arquivos EXR) e reutilizado no dataset de validação para garantir
    normalização consistente.
    """
    train_ds = LunarDataset(
        dataset_root, mode='train',
        group_size=group_size, val_per_group=val_per_group, random_seed=random_seed
    )
    val_ds = LunarDataset(
        dataset_root, mode='val',
        group_size=group_size, val_per_group=val_per_group, random_seed=random_seed,
        max_dim_km=train_ds.MAX_DIM_KM
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

# --- Teste do Dataloader ---
if __name__ == "__main__":
    # Ajuste o caminho para testar
    ROOT = "./dataset/LunarLanding_Dataset" 
    
    if not os.path.exists(ROOT):
        print(f"Caminho {ROOT} não existe. Crie a pasta ou ajuste o path.")
    else:
        tr_loader, val_loader = get_dataloaders(ROOT, batch_size=2)
        
        # Pega um batch
        imgs, targets = next(iter(tr_loader))

        print(f"✅ Dataloader Funcionando!")
        print(f"   Batch Imagem Shape:  {imgs.shape}")    # Esperado: [2, 1, H, W]
        print(f"   Batch Target Shape:  {targets.shape}") # Esperado: [2, 6]
        print(f"   Colunas: [xc, yc, zc, width, height, alt]")
        print(f"   xc   ∈ [{targets[:,0].min():.3f}, {targets[:,0].max():.3f}]  (ideal: [-1, 1])")
        print(f"   yc   ∈ [{targets[:,1].min():.3f}, {targets[:,1].max():.3f}]  (ideal: [-1, 1])")
        print(f"   zc   ∈ [{targets[:,2].min():.3f}, {targets[:,2].max():.3f}]  (ideal: [-1, 1])")
        print(f"   w    ∈ [{targets[:,3].min():.3f}, {targets[:,3].max():.3f}]  (ideal: [0, 1])")
        print(f"   h    ∈ [{targets[:,4].min():.3f}, {targets[:,4].max():.3f}]  (ideal: [0, 1])")
        print(f"   alt  ∈ [{targets[:,5].min():.3f}, {targets[:,5].max():.3f}]  (ideal: [0, 1])")
        print(f"   MAX_DIM_KM (treino) = {tr_loader.dataset.MAX_DIM_KM:.2f} km")