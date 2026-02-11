import torch
import torch.nn as nn

class LunarNavigationLoss(nn.Module):
    def __init__(self, lat_weight=1.0, lon_weight=1.0, alt_weight=1.0):
        """
        Função de Perda Híbrida para Navegação Lunar.
        
        Args:
            lat_weight (float): Peso para o erro da Latitude.
            lon_weight (float): Peso para o erro da Longitude.
            alt_weight (float): Peso para o erro da Altitude.
        """
        super(LunarNavigationLoss, self).__init__()
        self.lat_weight = lat_weight
        self.lon_weight = lon_weight
        self.alt_weight = alt_weight
        
        # Usamos L1 (Mean Absolute Error) para Lat e Alt pois é menos 
        # sensível a outliers do que o MSE no início do treino.
        self.regression_criterion = nn.L1Loss() 

    def cyclic_loss(self, pred, target, span=360.0):
        """
        Calcula a perda cíclica para a Longitude.
        Transforma a diferença em um componente angular.
        
        Math: Loss = 1 - cos(2 * pi * (pred - target) / span)
        Isso garante que 0 e 360 tenham erro 0.
        """
        diff = pred - target
        
        # Converte a diferença linear para radianos baseados no range (360)
        # O fator 2*pi normaliza o span para um ciclo completo trigonométrico.
        rad_diff = (diff / span) * 2 * torch.pi
        
        # 1 - cos(x) varia de 0 (sem erro) a 2 (erro máximo em 180 graus de diferença)
        loss = 1.0 - torch.cos(rad_diff)
        
        return torch.mean(loss)

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor [Batch, 3, H, W] -> Canais: (Lat, Lon, Alt)
            targets: Tensor [Batch, 3, H, W] -> Canais: (Lat, Lon, Alt)
            
        Nota: Este loss assume que os dados entram na escala REAL (graus e km)
        ou que o 'span' na cyclic_loss seja ajustado se os dados estiverem normalizados.
        
        Aqui, assumimos que você fará a desnormalização antes do loss 
        OU que o modelo já prevê na escala correta. 
        Se seus dados estiverem normalizados entre [-1, 1], ajuste o 'span' para 2.0.
        """
        
        # 1. Separar os canais
        lat_pred, lon_pred, alt_pred = preds[:, 0], preds[:, 1], preds[:, 2]
        lat_target, lon_target, alt_target = targets[:, 0], targets[:, 1], targets[:, 2]

        # 2. Latitude Loss (-60 a 60)
        # Não é cíclica, usa regressão padrão.
        loss_lat = self.regression_criterion(lat_pred, lat_target)

        # 3. Longitude Loss (0 a 360)
        # Cíclica: 0 deve ser perto de 360.
        # Se seus dados estão normalizados [-1, 1], mude span=360 para span=2
        loss_lon = self.cyclic_loss(lon_pred, lon_target, span=360.0)

        # 4. Altitude Loss
        # Regressão padrão
        loss_alt = self.regression_criterion(alt_pred, alt_target)

        # 5. Soma Ponderada
        total_loss = (loss_lat * self.lat_weight) + \
                     (loss_lon * self.lon_weight) + \
                     (loss_alt * self.alt_weight)

        return total_loss, (loss_lat, loss_lon, loss_alt)

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Simula Batch=2, Canais=3, 512x512
    # Canal 1 (Lon) simulando o problema da borda: Pred=1 grau, Target=359 graus
    
    # Criando dados dummy
    pred = torch.zeros(2, 3, 512, 512)
    target = torch.zeros(2, 3, 512, 512)
    
    # Caso crítico: Prediz 1°, Alvo é 359° (Erro real deve ser pequeno: 2°)
    pred[:, 1, :, :] = 1.0   
    target[:, 1, :, :] = 359.0 
    
    criterion = LunarNavigationLoss()
    
    loss, components = criterion(pred, target)
    
    print(f"Loss Total: {loss.item():.6f}")
    print(f"Componente Lat: {components[0].item():.6f}")
    print(f"Componente Lon (Cíclico): {components[1].item():.6f}") 
    print(f"Componente Alt: {components[2].item():.6f}")
    
    # O componente Lon deve ser muito baixo (próximo de 0), 
    # se fosse MSE comum seria enorme ((359-1)^2 = 128164).
    # Com 1-cos(diff), o erro para 2 graus é minúsculo (~0.0006).