import torch
import numpy as np

# Constantes Lunares
MOON_RADIUS_KM = 1737.4
ALT_MAX_KM     = 120.0   # Igual ao ALT_MAX_KM do dataload.py

# -------------------------------------------------------------------------
# Formato do tensor de saída (B, 6):
#   índice  | valor         | normalização  | range esperado
#   --------|---------------|---------------|------------------------------
#     0     | xc (km)       | / 1737.4      | [-1,  1]  (tanh)
#     1     | yc (km)       | / 1737.4      | [-1,  1]  (tanh)
#     2     | zc (km)       | / 1737.4      | [-1,  1]  (tanh)
#     3     | width  (km)   | / MAX_DIM_KM  | [ 0,  1]  (sigmoid)
#     4     | height (km)   | / MAX_DIM_KM  | [ 0,  1]  (sigmoid)
#     5     | altitude (km) | / 120.0       | [ 0,  1]  (sigmoid)
# -------------------------------------------------------------------------

def denormalize(tensor_pred, max_dim_km, alt_norm=ALT_MAX_KM):
    """
    Reverte a normalização da saída da rede.

    Args:
        tensor_pred: Tensor (B, 6) normalizado.
        max_dim_km (float): MAX_DIM_KM usado na normalização do dataset.
        alt_norm (float): Fator de normalização da altitude (default 120.0).
    Returns:
        Tensor (B, 6) com valores em km:
            [xc, yc, zc, width, height, alt]
    """
    pred = tensor_pred.clone()
    pred[:, :3]  = pred[:, :3]  * MOON_RADIUS_KM  # XYZ do centro
    pred[:, 3:5] = pred[:, 3:5] * max_dim_km       # Largura e altura
    pred[:, 5]   = pred[:, 5]   * alt_norm          # Altitude
    return pred


def cartesian_to_latlon(x, y, z):
    """
    Converte coordenadas cartesianas (km) para geográficas.
    Args: x, y, z – tensors com coordenadas em km (ponto NA SUPERFÍCIE, r ≈ 1737.4 km)
    Returns: lat (graus), lon (graus)
    """
    r = torch.sqrt(x**2 + y**2 + z**2).clamp(min=1e-8)

    lat_rad = torch.asin(torch.clamp(z / r, -1.0, 1.0))
    lat     = torch.rad2deg(lat_rad)

    lon_rad = torch.atan2(y, x)
    lon     = torch.rad2deg(lon_rad)
    lon     = torch.where(lon < 0, lon + 360.0, lon)

    return lat, lon


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em km entre dois pontos na superfície lunar.
    Entradas em GRAUS. Retorna tensor de distâncias em km.
    """
    lat1_r = torch.deg2rad(lat1)
    lon1_r = torch.deg2rad(lon1)
    lat2_r = torch.deg2rad(lat2)
    lon2_r = torch.deg2rad(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a  = torch.sin(dlat / 2)**2 + torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlon / 2)**2
    c  = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))

    return MOON_RADIUS_KM * c


def compute_metrics(preds, targets, max_dim_km, threshold_km=1.0):
    """
    Calcula métricas de performance para um batch.

    Args:
        preds      (Tensor): Saída normalizada da rede [B, 6].
        targets    (Tensor): Ground truth normalizado   [B, 6].
        max_dim_km (float):  MAX_DIM_KM do dataset (necessário para desnormalizar
                             largura e altura).
        threshold_km (float): Distância do centro abaixo da qual é "acerto".

    Returns:
        dict: Métricas médias do batch.
    """
    # 1. Desnormalizar
    pred_phys = denormalize(preds,   max_dim_km)  # (B, 6) em km
    targ_phys = denormalize(targets, max_dim_km)  # (B, 6) em km

    # 2. XYZ do centro → lat/lon (na superfície)
    pred_lat, pred_lon = cartesian_to_latlon(pred_phys[:, 0], pred_phys[:, 1], pred_phys[:, 2])
    targ_lat, targ_lon = cartesian_to_latlon(targ_phys[:, 0], targ_phys[:, 1], targ_phys[:, 2])

    # 3. Erros
    center_error  = haversine_distance(pred_lat, pred_lon, targ_lat, targ_lon)  # (B,) km
    width_error   = torch.abs(pred_phys[:, 3] - targ_phys[:, 3])               # (B,) km
    height_error  = torch.abs(pred_phys[:, 4] - targ_phys[:, 4])               # (B,) km
    alt_error     = torch.abs(pred_phys[:, 5] - targ_phys[:, 5])               # (B,) km

    # 4. Acurácia: % de amostras com erro do centro < threshold
    accuracy = (center_error < threshold_km).float().mean().item() * 100.0

    metrics = {
        "mean_center_error_km":  center_error.mean().item(),
        "mean_width_error_km":   width_error.mean().item(),
        "mean_height_error_km":  height_error.mean().item(),
        "mean_altitude_error_km": alt_error.mean().item(),
        # Alias mantido para compatibilidade com train.py
        "mean_position_error_km": center_error.mean().item(),
        "accuracy_percent":       accuracy,
        "threshold_used_km":      threshold_km,
    }

    return metrics


# --- Bloco de Teste ---
if __name__ == "__main__":
    print("🧪 Testando compute_metrics com formato [xc, yc, zc, width, height, alt]...")

    MAX_DIM_KM_TEST = 300.0   # valor fictício para o teste

    # Centro fictício: lat=0°, lon=0°, na superfície
    #   x = 1737.4, y = 0, z = 0  →  xc_n = 1.0, yc_n = 0.0, zc_n = 0.0
    xc_n = 1.0
    # width=150km, height=100km, alt=50km  →  normalizados:
    w_n  = 150.0 / MAX_DIM_KM_TEST
    h_n  = 100.0 / MAX_DIM_KM_TEST
    a_n  = 50.0  / ALT_MAX_KM

    target_tensor = torch.tensor([[xc_n, 0.0, 0.0, w_n, h_n, a_n],
                                   [xc_n, 0.0, 0.0, w_n, h_n, a_n]])

    # Pred com pequeno desvio no centro
    pred_tensor = target_tensor.clone()
    pred_tensor[:, 0] -= 0.0005   # pequeno desvio em x

    results = compute_metrics(pred_tensor, target_tensor, max_dim_km=MAX_DIM_KM_TEST, threshold_km=1.0)

    print(f"\nResultados Simulados:")
    print(f"  🎯 Threshold: {results['threshold_used_km']} km")
    print(f"  📍 Erro Centro:    {results['mean_center_error_km']:.4f} km")
    print(f"  ↔  Erro Largura:   {results['mean_width_error_km']:.4f} km")
    print(f"  ↕  Erro Altura:    {results['mean_height_error_km']:.4f} km")
    print(f"  🏔  Erro Altitude:  {results['mean_altitude_error_km']:.4f} km")
    print(f"  ✅ Acurácia:       {results['accuracy_percent']:.2f}%")