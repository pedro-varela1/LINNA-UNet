import os
import glob
import cv2
import torch
import numpy as np
import OpenEXR
import Imath
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constante do raio da Lua em km
MOON_RADIUS_KM = 1737.4

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
    def __init__(self, root_dir, transform=None, mode='train', val_split=0.1):
        """
        Args:
            root_dir (str): Caminho raiz do dataset.
            transform (callable, optional): Transformações de data augmentation.
            mode (str): 'train' ou 'val'.
            val_split (float): Porcentagem de dados para validação.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Criar augmentations se em modo de treino
        if mode == 'train':
            self.augmentations = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(per_channel=False, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ])
        else:
            self.augmentations = None
        
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

        # 2. Divisão Treino/Validação
        indices = list(range(len(self.img_files)))
        train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)
        
        self.active_indices = train_idx if mode == 'train' else val_idx

        # 3. Constantes de Normalização para Coordenadas Cartesianas
        # As coordenadas x, y, z estarão aproximadamente no range:
        # Para alt ~0-120km e raio_lua=1737.4km, r total ~1737-1857km
        # Normalizamos x, y, z dividindo por um valor de escala apropriado
        # Usaremos raio_lua + altitude_max como escala
        self.NORM_RADIUS = MOON_RADIUS_KM + 120.0  # ~1857.4 km
        # Com isso, x, y, z normalizados ficarão aproximadamente em [-1, 1]

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        # Mapeia o índice do dataset para o índice real do arquivo
        real_idx = self.active_indices[idx]
        
        # --- A. Carregar Input (Imagem) ---
        img_path = self.img_files[real_idx]
        # Carrega em Grayscale (H, W)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # --- B. Carregar Target (Coordenadas) ---
        # 1. Lat/Lon do arquivo EXR
        exr_path = self.exr_files[real_idx]
        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        r_str = exr_file.channel('R', FLOAT)
        g_str = exr_file.channel('G', FLOAT)
        lat_map = np.frombuffer(r_str, dtype=np.float32).reshape(size[1], size[0])
        lon_map = np.frombuffer(g_str, dtype=np.float32).reshape(size[1], size[0])
        
        # 2. Altitude do arquivo TXT
        alt_path = self.alt_files[real_idx]
        with open(alt_path, 'r') as f:
            altitude_scalar = float(f.read().strip())
        
        # --- C. Aplicar Augmentations (se em treino) ---
        # Albumentations espera imagem em (H, W) ou (H, W, C)
        if self.augmentations is not None:
            # Empilhar lat, lon como canais adicionais para serem transformados juntos
            # Isso garante que augmentations geométricas sejam aplicadas consistentemente
            stacked = np.stack([image, lat_map, lon_map], axis=-1).astype(np.float32)
            augmented = self.augmentations(image=stacked)
            stacked_aug = augmented['image']
            image = stacked_aug[:, :, 0]
            lat_map = stacked_aug[:, :, 1]
            lon_map = stacked_aug[:, :, 2]
        
        # Normalizar imagem [0, 255] -> [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # --- D. Extrair Coordenadas das 4 Extremidades ---
        H, W = lat_map.shape
        # Cantos: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        corners = [
            (0, 0),           # Top-Left
            (0, W-1),         # Top-Right
            (H-1, 0),         # Bottom-Left
            (H-1, W-1)        # Bottom-Right
        ]
        
        coords_list = []
        for (y, x) in corners:
            lat = lat_map[y, x]
            lon = lon_map[y, x]
            alt = altitude_scalar
            
            # Converter para coordenadas cartesianas
            x_cart, y_cart, z_cart = latlon_to_cartesian(lat, lon, alt)
            
            # Normalizar
            x_norm = x_cart / self.NORM_RADIUS
            y_norm = y_cart / self.NORM_RADIUS
            z_norm = z_cart / self.NORM_RADIUS
            
            coords_list.extend([x_norm, y_norm, z_norm])
        
        # Target: vetor com 12 valores [x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4]
        target = np.array(coords_list, dtype=np.float32)
        
        # Adicionar dimensão de canal à imagem -> (1, H, W)
        image = np.expand_dims(image, axis=0)

        # Converter para Tensores PyTorch
        return torch.from_numpy(image), torch.from_numpy(target)

    @staticmethod
    def denormalize(tensor_pred):
        """
        Função utilitária para converter a saída da rede (coordenadas cartesianas normalizadas)
        de volta para coordenadas cartesianas em km.
        Args: tensor_pred (B, 12) - [x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4]
        Returns: tensor com (B, 12) - coordenadas em km
        """
        NORM_RADIUS = MOON_RADIUS_KM + 120.0  # ~1857.4 km
        
        pred = tensor_pred.clone()  # Evita modificar in-place
        
        # Desnormalizar todas as coordenadas
        pred = pred * NORM_RADIUS
        
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

def get_dataloaders(dataset_root, batch_size=4, val_split=0.1, num_workers=2):
    """
    Helper para criar os DataLoaders de treino e validação.
    """
    train_ds = LunarDataset(dataset_root, mode='train', val_split=val_split)
    val_ds = LunarDataset(dataset_root, mode='val', val_split=val_split)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
        print(f"   Batch Imagem Shape: {imgs.shape}")   # Esperado: [2, 1, 512, 512]
        print(f"   Batch Target Shape: {targets.shape}") # Esperado: [2, 12]
        
        # Verifica Ranges Normalizados (agora são 4 pontos com 3 coordenadas cada)
        print(f"   Target Min: {targets.min().item():.3f}")
        print(f"   Target Max: {targets.max().item():.3f}")
        print(f"   (Ideal: entre -1 e 1 para coordenadas esféricas normalizadas)")