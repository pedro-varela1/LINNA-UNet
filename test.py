import torch
import numpy as np

# Constantes Lunares
MOON_RADIUS_KM = 1737.4
NORM_RADIUS = MOON_RADIUS_KM + 120.0  # ~1857.4 km

def denormalize_cartesian(tensor):
    """
    Reverte a normalização das coordenadas cartesianas.
    Args: tensor (B, 12) - [x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4] normalizados
    Returns: tensor (B, 12) com coordenadas em km
    """
    tensor = tensor.clone()
    return tensor * NORM_RADIUS

def cartesian_to_latlon(x, y, z):
    """
    Converte coordenadas cartesianas (km) para geográficas.
    Args: x, y, z tensors com coordenadas em km
    Returns: lat (graus), lon (graus), alt (km) como tensors
    """
    # Calcular raio total
    r = torch.sqrt(x**2 + y**2 + z**2)
    
    # Altitude = raio total - raio da lua
    alt = r - MOON_RADIUS_KM
    
    # Latitude (em radianos, depois converter para graus)
    lat_rad = torch.asin(torch.clamp(z / r, -1.0, 1.0))
    lat = torch.rad2deg(lat_rad)
    
    # Longitude (em radianos, depois converter para graus)
    lon_rad = torch.atan2(y, x)
    lon = torch.rad2deg(lon_rad)
    
    # Garantir longitude no range [0, 360]
    lon = torch.where(lon < 0, lon + 360, lon)
    
    return lat, lon, alt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em km entre dois pontos na superfície lunar.
    Entrada deve estar em GRAUS.
    Retorna tensor de distâncias em KM.
    """
    # Converter graus para radianos
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Fórmula de Haversine
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    distance_km = MOON_RADIUS_KM * c
    return distance_km

def compute_metrics(preds, targets, threshold_km=1.0):
    """
    Calcula métricas de performance para um batch com 4 pontos.
    
    Args:
        preds (Tensor): Saída da rede normalizada [B, 12] - 4 pontos × 3 coordenadas
        targets (Tensor): Ground truth normalizado [B, 12]
        threshold_km (float): Distância limite para considerar um "acerto" (ex: 1km).
        
    Returns:
        dict: Dicionário com métricas médias do batch.
    """
    batch_size = preds.shape[0]
    
    # 1. Desnormalizar para coordenadas cartesianas em km
    pred_cart = denormalize_cartesian(preds)  # (B, 12)
    targ_cart = denormalize_cartesian(targets)  # (B, 12)
    
    # 2. Separar os 4 pontos
    # Reshape: (B, 12) -> (B, 4, 3) onde dim 2 = [x, y, z]
    pred_cart = pred_cart.view(batch_size, 4, 3)
    targ_cart = targ_cart.view(batch_size, 4, 3)
    
    # 3. Calcular erros para cada ponto
    position_errors = []
    altitude_errors = []
    
    for i in range(4):  # Para cada um dos 4 pontos
        # Extrair coordenadas do ponto i
        pred_x, pred_y, pred_z = pred_cart[:, i, 0], pred_cart[:, i, 1], pred_cart[:, i, 2]
        targ_x, targ_y, targ_z = targ_cart[:, i, 0], targ_cart[:, i, 1], targ_cart[:, i, 2]
        
        # Converter para lat/lon/alt
        pred_lat, pred_lon, pred_alt = cartesian_to_latlon(pred_x, pred_y, pred_z)
        targ_lat, targ_lon, targ_alt = cartesian_to_latlon(targ_x, targ_y, targ_z)
        
        # Erro de posição (Haversine)
        pos_error = haversine_distance(pred_lat, pred_lon, targ_lat, targ_lon)
        position_errors.append(pos_error)
        
        # Erro de altitude
        alt_error = torch.abs(pred_alt - targ_alt)
        altitude_errors.append(alt_error)
    
    # Stack todos os erros: (4, B) -> (B, 4)
    position_errors = torch.stack(position_errors, dim=1)  # (B, 4)
    altitude_errors = torch.stack(altitude_errors, dim=1)  # (B, 4)
    
    # 4. Métricas agregadas (média sobre todos os pontos)
    mean_pos_error = position_errors.mean().item()
    mean_alt_error = altitude_errors.mean().item()
    
    # Acurácia: % de pontos com erro < threshold
    accurate_points = (position_errors < threshold_km).float().mean().item() * 100.0
    
    metrics = {
        "mean_position_error_km": mean_pos_error,
        "median_position_error_km": position_errors.median().item(),
        "mean_altitude_error_km": mean_alt_error,
        "accuracy_percent": accurate_points,
        "threshold_used_km": threshold_km
    }

    return metrics

# --- Bloco de Teste ---
if __name__ == "__main__":
    # Simulação de Teste
    print("🧪 Testando cálculo de métricas com 4 pontos...")
    
    # Criar dados fake (Batch=2, 12 valores cada)
    # Para latitude 0, longitude 0, altitude 50km:
    # x = (1737.4 + 50) * cos(0) * cos(0) = 1787.4
    # y = 0, z = 0
    # Normalizado: x_norm = 1787.4 / 1857.4 ≈ 0.962
    
    target_tensor = torch.zeros(2, 12)
    # 4 pontos com mesmas coordenadas para simplificar
    for i in range(4):
        target_tensor[:, i*3] = 1787.4 / NORM_RADIUS  # X normalizado
        target_tensor[:, i*3 + 1] = 0.0  # Y normalizado
        target_tensor[:, i*3 + 2] = 0.0  # Z normalizado
    
    # Pred: pequeno erro nas coordenadas
    pred_tensor = target_tensor.clone()
    pred_tensor[:, 0] += 0.001  # Pequeno deslocamento no primeiro ponto
    
    # Calcula
    results = compute_metrics(pred_tensor, target_tensor, threshold_km=1.0)
    
    print(f"\nResultados Simulados:")
    print(f"🎯 Threshold de Acerto: {results['threshold_used_km']} km")
    print(f"📏 Erro Médio de Posição: {results['mean_position_error_km']:.4f} km")
    print(f"📈 Acurácia (< 1km): {results['accuracy_percent']:.2f}%")
    print(f"🏔️  Erro Médio de Altitude: {results['mean_altitude_error_km']:.4f} km")