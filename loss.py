import torch
import torch.nn as nn
import math

# -------------------------------------------------------------------------
# Formato do tensor de entrada (B, 6):
#   [0:3]  xc_n, yc_n, zc_n  ∈ [-1, 1]  — centro na superfície / MOON_RADIUS_KM
#   [3]    width_n           ∈ [ 0, 1]  — largura / MAX_DIM_KM
#   [4]    height_n          ∈ [ 0, 1]  — altura  / MAX_DIM_KM
#   [5]    alt_n             ∈ [ 0, 1]  — altitude / ALT_MAX_KM
# -------------------------------------------------------------------------


class LunarNavigationLoss(nn.Module):
    """
    Função de Perda para Navegação Lunar com saída (B, 6).

    Componentes:
      1. Distância superficial normalizada (centro XYZ)
         Os vetores [xc_n, yc_n, zc_n] são unitários (ponto na superfície
         dividido por MOON_RADIUS_KM), então a distância angular é:

             theta = arccos( clamp(p . q, -1, 1) )

         Normalizada por pi para ficar em [0, 1]:

             L_center = mean( theta / pi )

      2. Erro L1 normalizado para largura e altura do footprint:

             L_dim = mean( |width_n_pred - width_n_gt|
                         + |height_n_pred - height_n_gt| ) / 2

      3. Erro L1 normalizado para altitude:

             L_alt = mean( |alt_n_pred - alt_n_gt| )

      Total = w_center * L_center + w_dim * L_dim + w_alt * L_alt
    """

    def __init__(self, w_center=1.0, w_dim=0.5, w_alt=0.5):
        """
        Args:
            w_center (float): Peso da distância superficial do centro.
            w_dim    (float): Peso do erro de largura/altura.
            w_alt    (float): Peso do erro de altitude.
        """
        super().__init__()
        self.w_center = w_center
        self.w_dim    = w_dim
        self.w_alt    = w_alt

    def forward(self, preds, targets):
        """
        Args:
            preds   (Tensor): (B, 6) — saída normalizada da rede.
            targets (Tensor): (B, 6) — ground truth normalizado.
        Returns:
            loss (Tensor): escalar diferenciável.
            components (tuple): (L_center, L_dim, L_alt) para logging.
        """
        # ----- 1. Distância superficial normalizada -----
        p_xyz = preds[:,   :3]   # (B, 3)
        t_xyz = targets[:, :3]   # (B, 3)

        # Normalizar para vetores unitários antes do produto interno.
        # O GT já tem norma = 1 por construção (ver dataload), mas a rede pode
        # prever vetores com norma != 1 — sem normalizar o ponto estaria fora
        # da superfície lunar, tornando a distância angular incorreta.
        p_xyz = torch.nn.functional.normalize(p_xyz, p=2, dim=1)
        t_xyz = torch.nn.functional.normalize(t_xyz, p=2, dim=1)

        # Produto interno → cos(theta), clamp por estabilidade numérica
        dot = (p_xyz * t_xyz).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Distância angular em [0, pi], normalizada para [0, 1]
        theta_norm = torch.acos(dot) / math.pi   # (B,)
        L_center   = theta_norm.mean()

        # ----- 2. Erro de dimensões (footprint) -----
        L_dim = (
            (preds[:, 3] - targets[:, 3]).abs() +
            (preds[:, 4] - targets[:, 4]).abs()
        ).mean() / 2.0

        # ----- 3. Erro de altitude -----
        L_alt = (preds[:, 5] - targets[:, 5]).abs().mean()

        # ----- Total ponderado -----
        loss = self.w_center * L_center + self.w_dim * L_dim + self.w_alt * L_alt

        return loss, (L_center, L_dim, L_alt)


# --- Bloco de Teste ---
if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4

    # Ponto de referência: lat=0, lon=0 na superfície -> xc_n=1, yc_n=0, zc_n=0
    targets = torch.zeros(B, 6)
    targets[:, 0] = 1.0   # xc_n = 1  (polo lon=0)
    targets[:, 3] = 0.5   # width_n
    targets[:, 4] = 0.4   # height_n
    targets[:, 5] = 0.3   # alt_n

    # Pred com pequeno desvio
    preds = targets.clone()
    preds[:, 0] = 0.9999  # ~0.81° de distância angular
    preds[:, 1] = 0.01
    preds[:, 3] += 0.05
    preds[:, 5] -= 0.02

    criterion = LunarNavigationLoss()
    loss, (lc, ld, la) = criterion(preds, targets)

    print(f"Loss Total  : {loss.item():.6f}")
    print(f"  L_center  : {lc.item():.6f}  (dist superficial normalizada, 0=perfeito)")
    print(f"  L_dim     : {ld.item():.6f}  (erro largura/altura normalizado)")
    print(f"  L_alt     : {la.item():.6f}  (erro altitude normalizado)")
