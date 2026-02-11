import torch
import torch.optim as optim
import csv
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Importando seus módulos customizados
# Certifique-se de que os arquivos estão na mesma pasta
from unet import LunarUNet
from dataload import get_dataloaders, LunarDataset, MOON_RADIUS_KM, cartesian_to_latlon
from test import compute_metrics

# --- CONFIGURAÇÕES E HIPERPARÂMETROS ---
CONFIG = {
    "dataset_root": "./dataset/LunarLanding_Dataset",
    "num_epochs": 50,
    "batch_size": 2,           # Ajuste conforme a VRAM da sua GPU
    "learning_rate": 1e-4,     # Learning rate padrão para U-Net/Adam
    "val_split": 0.1,
    "save_dir": "./checkpoints",
    "csv_log_file": "training_metrics.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "metric_threshold_km": 1.0,
    "viz_dir": "./train_visualizations",
}

def create_sphere(radius, resolution=30):
    """Cria uma malha esférica para visualização."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    return x, y, z

def visualize_predictions(model, val_loader, epoch, device, save_dir, num_samples=3):
    """
    Visualiza predições do modelo em uma esfera lunar.
    Ground truth em verde, predições em vermelho.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # Pegar alguns exemplos
    samples_collected = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if samples_collected >= num_samples:
                break
            
            images = images.to(device)
            preds = model(images)
            
            # Processar cada imagem no batch
            for i in range(min(images.shape[0], num_samples - samples_collected)):
                # Desnormalizar
                pred_denorm = LunarDataset.denormalize(preds[i:i+1])  # (1, 12)
                targ_denorm = LunarDataset.denormalize(targets[i:i+1])  # (1, 12)
                
                # Reshape para (4, 3)
                pred_coords = pred_denorm.view(4, 3).cpu().numpy()
                targ_coords = targ_denorm.view(4, 3).cpu().numpy()
                
                # Criar figura
                fig = plt.figure(figsize=(14, 6))
                
                # Subplot 1: Imagem original
                ax1 = fig.add_subplot(1, 2, 1)
                img_np = images[i, 0].cpu().numpy()
                ax1.imshow(img_np, cmap='gray')
                ax1.set_title(f'Input Image - Epoch {epoch}', fontsize=12, fontweight='bold')
                ax1.axis('off')
                
                # Subplot 2: Esfera 3D
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                
                # Desenhar esfera base
                sphere_radius = MOON_RADIUS_KM
                x_sphere, y_sphere, z_sphere = create_sphere(sphere_radius, resolution=40)
                ax2.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', 
                               alpha=0.2, edgecolor='none')
                
                # Plotar pontos do ground truth (verde)
                for j in range(4):
                    x_gt, y_gt, z_gt = targ_coords[j]
                    ax2.scatter(x_gt, y_gt, z_gt, c='green', s=100, marker='o', 
                              label='Ground Truth' if j == 0 else '', edgecolors='darkgreen', linewidth=2)
                
                # Plotar pontos preditos (vermelho)
                for j in range(4):
                    x_pr, y_pr, z_pr = pred_coords[j]
                    ax2.scatter(x_pr, y_pr, z_pr, c='red', s=100, marker='^', 
                              label='Prediction' if j == 0 else '', edgecolors='darkred', linewidth=2)
                
                # Conectar os 4 pontos para formar um quadrilátero (GT)
                corner_order = [0, 1, 3, 2, 0]  # TL, TR, BR, BL, TL
                gt_x = [targ_coords[idx, 0] for idx in corner_order]
                gt_y = [targ_coords[idx, 1] for idx in corner_order]
                gt_z = [targ_coords[idx, 2] for idx in corner_order]
                ax2.plot(gt_x, gt_y, gt_z, 'g-', linewidth=2, alpha=0.6)
                
                # Conectar os 4 pontos preditos
                pr_x = [pred_coords[idx, 0] for idx in corner_order]
                pr_y = [pred_coords[idx, 1] for idx in corner_order]
                pr_z = [pred_coords[idx, 2] for idx in corner_order]
                ax2.plot(pr_x, pr_y, pr_z, 'r--', linewidth=2, alpha=0.6)
                
                # Configurar visualização
                max_range = sphere_radius * 1.3
                ax2.set_xlim([-max_range, max_range])
                ax2.set_ylim([-max_range, max_range])
                ax2.set_zlim([-max_range, max_range])
                ax2.set_xlabel('X (km)', fontweight='bold')
                ax2.set_ylabel('Y (km)', fontweight='bold')
                ax2.set_zlabel('Z (km)', fontweight='bold')
                ax2.set_title(f'4 Corner Points - Epoch {epoch}', fontsize=12, fontweight='bold')
                ax2.legend(loc='upper right')
                ax2.set_box_aspect([1, 1, 1])
                ax2.view_init(elev=20, azim=45)
                
                plt.tight_layout()
                
                # Salvar
                save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_sample_{samples_collected + 1}.png')
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                samples_collected += 1
                if samples_collected >= num_samples:
                    break
    
    print(f"   📊 Visualizações salvas em: {save_dir}")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # Acumuladores de métricas físicas
    total_pos_error = 0.0
    total_alt_error = 0.0
    total_accuracy = 0.0
    
    pbar = tqdm(loader, desc="Training", unit="batch")
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # 1. Forward
        optimizer.zero_grad()
        preds = model(images)
        
        # 2. Cálculo da Loss (MSE direto nos valores normalizados)
        loss = criterion(preds, targets)
        
        # 3. Backward
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 4. Métricas
        with torch.no_grad():
            batch_metrics = compute_metrics(preds, targets, threshold_km=CONFIG["metric_threshold_km"])
            total_pos_error += batch_metrics["mean_position_error_km"]
            total_alt_error += batch_metrics["mean_altitude_error_km"]
            total_accuracy  += batch_metrics["accuracy_percent"]
            
        pbar.set_postfix(loss=loss.item())

    # Médias da época
    avg_loss = running_loss / len(loader)
    avg_pos_err = total_pos_error / len(loader)
    avg_alt_err = total_alt_error / len(loader)
    avg_acc = total_accuracy / len(loader)
    
    return avg_loss, avg_pos_err, avg_alt_err, avg_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_pos_error = 0.0
    total_alt_error = 0.0
    total_accuracy = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", unit="batch"):
            images = images.to(device)
            targets = targets.to(device)
            
            preds = model(images)
            
            # Cálculo da Loss (MSE)
            loss = criterion(preds, targets)
            running_loss += loss.item()
            
            # Métricas Físicas
            batch_metrics = compute_metrics(preds, targets, threshold_km=CONFIG["metric_threshold_km"])
            total_pos_error += batch_metrics["mean_position_error_km"]
            total_alt_error += batch_metrics["mean_altitude_error_km"]
            total_accuracy  += batch_metrics["accuracy_percent"]

    avg_loss = running_loss / len(loader)
    avg_pos_err = total_pos_error / len(loader)
    avg_alt_err = total_alt_error / len(loader)
    avg_acc = total_accuracy / len(loader)
    
    return avg_loss, avg_pos_err, avg_alt_err, avg_acc

def main():
    # 1. Setup de Diretórios e CSV
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    os.makedirs(CONFIG["viz_dir"], exist_ok=True)
    
    # Criar/Sobrescrever CSV e escrever cabeçalho
    with open(CONFIG["csv_log_file"], mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Train_Loss", "Train_Pos_Err_km", "Train_Alt_Err_km", "Train_Acc_%",
            "Val_Loss", "Val_Pos_Err_km", "Val_Alt_Err_km", "Val_Acc_%", "Time_Sec"
        ])

    print(f"🚀 Iniciando treinamento no dispositivo: {CONFIG['device']}")
    
    # 2. Carregar Dados
    train_loader, val_loader = get_dataloaders(
        CONFIG["dataset_root"], 
        batch_size=CONFIG["batch_size"], 
        val_split=CONFIG["val_split"],
        num_workers=CONFIG["num_workers"]
    )

    # 3. Inicializar Modelo e Otimização
    model = LunarUNet(n_channels=1, n_classes=12).to(CONFIG["device"])
    
    # Função de Perda - MSE já que os valores estão normalizados em [-1, 1]
    criterion = torch.nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Scheduler para reduzir o LR se o loss parar de cair
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    # 4. Loop de Treinamento
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        start_time = time.time()
        
        print(f"\n--- Epoch {epoch}/{CONFIG['num_epochs']} ---")
        
        # Treino
        tr_loss, tr_pos, tr_alt, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG["device"])
        
        # Validação
        val_loss, val_pos, val_alt, val_acc = validate(model, val_loader, criterion, CONFIG["device"])
        
        epoch_time = time.time() - start_time
        
        # Atualizar Scheduler
        scheduler.step(val_loss)

        # 5. Logging no CSV
        with open(CONFIG["csv_log_file"], mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{tr_loss:.4f}", f"{tr_pos:.4f}", f"{tr_alt:.4f}", f"{tr_acc:.2f}",
                f"{val_loss:.4f}", f"{val_pos:.4f}", f"{val_alt:.4f}", f"{val_acc:.2f}",
                f"{epoch_time:.2f}"
            ])
            
        print(f"   Train Loss: {tr_loss:.4f} | Pos Err: {tr_pos:.2f}km | Acc: {tr_acc:.1f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Pos Err: {val_pos:.2f}km | Acc: {val_acc:.1f}%")

        # 6. Criar visualizações ao final de cada época
        visualize_predictions(model, val_loader, epoch, CONFIG["device"], CONFIG["viz_dir"], num_samples=3)

        # 7. Salvar Checkpoint (Melhor Loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(CONFIG["save_dir"], "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   ⭐ Melhor modelo salvo em: {save_path}")

        # Opcional: Salvar último modelo
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "last_model.pth"))

    print("\n🏁 Treinamento finalizado!")

if __name__ == "__main__":
    main()