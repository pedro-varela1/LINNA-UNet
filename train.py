import torch
import torch.optim as optim
import csv
import os
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Importando seus módulos customizados
# Certifique-se de que os arquivos estão na mesma pasta
from unet import LunarUNet
from dataload import get_dataloaders, LunarDataset, MOON_RADIUS_KM, cartesian_to_latlon, ALT_MAX_KM
from loss import LunarNavigationLoss
from test import compute_metrics

# --- CONFIGURAÇÕES E HIPERPARÂMETROS ---
CONFIG = {
    "dataset_root": "../LINNA-Crater/LunarLanding_Dataset/LunarLanding_Dataset",
    "num_epochs": 50,
    "batch_size": 4,           # Ajuste conforme a VRAM da sua GPU
    "learning_rate": 1e-4,     # Learning rate padrão para U-Net/Adam
    "val_split": 0.1,          # mantido para referência (não usado — ver group_size/val_per_group)
    "group_size": 12,          # imagens por grupo/região
    "val_per_group": 2,        # imagens de validação por grupo (sorteadas aleatoriamente)
    "random_seed": 42,         # semente para reprodutibilidade do sorteio
    "save_dir": "./checkpoints",
    "csv_log_file": "training_metrics.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "metric_threshold_km": 1.0,
    "viz_dir": "./train_visualizations",
}

def _parse_replay(replay):
    """
    Extrai os nomes das augmentations que foram efetivamente aplicadas
    a partir do dict de replay do ReplayCompose.
    """
    applied = []
    if replay is None:
        return applied
    for t in replay.get('transforms', []):
        if t.get('applied', False):
            name = t.get('__class_fullname__', '')
            # Pega só o nome curto (sem módulo)
            short = name.split('.')[-1] if '.' in name else name
            applied.append(short)
    return applied


def visualize_dataset_samples(train_ds, val_ds, save_dir):
    """
    Gera uma figura com 3 exemplos antes do treino:
      - Coluna 0: sem augmentation (val_ds)
      - Coluna 1 e 2: com augmentation (train_ds)
    Labels em escala real:
      lat/lon do centro, altitude, largura e altura em km.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Coleta as amostras
    samples = []  # lista de (image_np, target_tensor, aug_names)

    # 1 sem augmentation (do val_ds, que tem augmentations=None)
    img_t, tgt_t = val_ds[0]
    samples.append((img_t[0].numpy(), tgt_t, []))

    # 2 com augmentation (do train_ds)
    for i in range(2):
        img_t, tgt_t = train_ds[i]
        replay = train_ds._last_aug_replay
        aug_names = _parse_replay(replay)
        samples.append((img_t[0].numpy(), tgt_t, aug_names))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    titles = ['Sem Augmentation (val)', 'Com Augmentation #1 (treino)', 'Com Augmentation #2 (treino)']

    for ax, (img_np, tgt, aug_names), title in zip(axes, samples, titles):
        ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')

        # Desnormalizar labels para escala real
        tgt_phys = val_ds.denormalize(tgt.unsqueeze(0))[0]  # (6,)
        xc_km, yc_km, zc_km = tgt_phys[:3].tolist()
        width_km  = tgt_phys[3].item()
        height_km = tgt_phys[4].item()
        alt_km    = tgt_phys[5].item()

        # Converter XYZ -> lat/lon
        r    = math.sqrt(xc_km**2 + yc_km**2 + zc_km**2)
        lat  = math.degrees(math.asin(max(-1.0, min(1.0, zc_km / r))))
        lon  = math.degrees(math.atan2(yc_km, xc_km))
        lon  = lon + 360.0 if lon < 0 else lon

        label_lines = [
            f"Lat:  {lat:+.3f}°",
            f"Lon:  {lon:.3f}°",
            f"Alt:  {alt_km:.1f} km",
            f"W:    {width_km:.1f} km",
            f"H:    {height_km:.1f} km",
        ]
        if aug_names:
            label_lines += ['', 'Augmentations:'] + [f"  • {a}" for a in aug_names]
        else:
            label_lines += ['', '(sem augmentation)']

        label_text = '\n'.join(label_lines)
        ax.text(
            0.02, 0.02, label_text,
            transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.65),
            color='white', family='monospace'
        )

    plt.suptitle('Exemplos do Dataset antes do Treino', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'dataset_samples_preview.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"   🖼️  Preview do dataset salvo em: {save_path}")


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
    Mostra o ponto central previsto vs. ground truth, com a indicação
    das dimensões do footprint.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    MAX_DIM_KM = val_loader.dataset.MAX_DIM_KM
    ALT_NORM   = val_loader.dataset.ALT_NORM

    samples_collected = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if samples_collected >= num_samples:
                break

            images = images.to(device)
            preds  = model(images)

            for i in range(min(images.shape[0], num_samples - samples_collected)):
                # Desnormalizar usando o método de instância do dataset
                ds = val_loader.dataset
                pred_phys = ds.denormalize(preds[i:i+1].cpu())    # (1, 6)
                targ_phys = ds.denormalize(targets[i:i+1].cpu())  # (1, 6)

                # Extrair componentes [xc, yc, zc, width, height, alt]
                p_xyz  = pred_phys[0, :3].numpy()
                p_wha  = pred_phys[0, 3:].numpy()   # [width, height, alt]
                t_xyz  = targ_phys[0, :3].numpy()
                t_wha  = targ_phys[0, 3:].numpy()

                fig = plt.figure(figsize=(14, 6))

                # Subplot 1: Imagem original
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.imshow(images[i, 0].cpu().numpy(), cmap='gray')
                ax1.set_title(f'Input Image — Epoch {epoch}', fontsize=12, fontweight='bold')
                ax1.axis('off')

                # Anotações na imagem
                ax1.set_xlabel(
                    f"GT   — centro:({t_xyz[0]:.0f}, {t_xyz[1]:.0f}, {t_xyz[2]:.0f}) km  "
                    f"| {t_wha[0]:.1f}×{t_wha[1]:.1f} km  | alt {t_wha[2]:.1f} km\n"
                    f"Pred — centro:({p_xyz[0]:.0f}, {p_xyz[1]:.0f}, {p_xyz[2]:.0f}) km  "
                    f"| {p_wha[0]:.1f}×{p_wha[1]:.1f} km  | alt {p_wha[2]:.1f} km",
                    fontsize=7
                )

                # Subplot 2: Esfera 3D — apenas o ponto central
                ax2 = fig.add_subplot(1, 2, 2, projection='3d')

                sphere_radius = MOON_RADIUS_KM
                x_s, y_s, z_s = create_sphere(sphere_radius, resolution=40)
                ax2.plot_surface(x_s, y_s, z_s, color='lightgray', alpha=0.2, edgecolor='none')

                # Ground truth — centro (verde)
                ax2.scatter(*t_xyz, c='green', s=150, marker='o',
                            label='GT Center', edgecolors='darkgreen', linewidth=2)

                # Predição — centro (vermelho)
                ax2.scatter(*p_xyz, c='red', s=150, marker='^',
                            label='Pred Center', edgecolors='darkred', linewidth=2)

                max_range = sphere_radius * 1.3
                ax2.set_xlim([-max_range, max_range])
                ax2.set_ylim([-max_range, max_range])
                ax2.set_zlim([-max_range, max_range])
                ax2.set_xlabel('X (km)', fontweight='bold')
                ax2.set_ylabel('Y (km)', fontweight='bold')
                ax2.set_zlabel('Z (km)', fontweight='bold')
                ax2.set_title(f'Center Point — Epoch {epoch}', fontsize=12, fontweight='bold')
                ax2.legend(loc='upper right')
                ax2.set_box_aspect([1, 1, 1])
                ax2.view_init(elev=20, azim=45)

                plt.tight_layout()

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

    total_pos_error = 0.0
    total_alt_error = 0.0
    total_accuracy  = 0.0

    max_dim_km = loader.dataset.MAX_DIM_KM

    pbar = tqdm(loader, desc="Training", unit="batch")

    for images, targets in pbar:
        images  = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)

        loss, _ = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        with torch.no_grad():
            batch_metrics = compute_metrics(
                preds, targets,
                max_dim_km=max_dim_km,
                threshold_km=CONFIG["metric_threshold_km"]
            )
            total_pos_error += batch_metrics["mean_position_error_km"]
            total_alt_error += batch_metrics["mean_altitude_error_km"]
            total_accuracy  += batch_metrics["accuracy_percent"]

        pbar.set_postfix(loss=loss.item())

    avg_loss    = running_loss    / len(loader)
    avg_pos_err = total_pos_error / len(loader)
    avg_alt_err = total_alt_error / len(loader)
    avg_acc     = total_accuracy  / len(loader)

    return avg_loss, avg_pos_err, avg_alt_err, avg_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss    = 0.0
    total_pos_error = 0.0
    total_alt_error = 0.0
    total_accuracy  = 0.0

    max_dim_km = loader.dataset.MAX_DIM_KM

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", unit="batch"):
            images  = images.to(device)
            targets = targets.to(device)

            preds = model(images)
            loss, _  = criterion(preds, targets)
            running_loss += loss.item()

            batch_metrics = compute_metrics(
                preds, targets,
                max_dim_km=max_dim_km,
                threshold_km=CONFIG["metric_threshold_km"]
            )
            total_pos_error += batch_metrics["mean_position_error_km"]
            total_alt_error += batch_metrics["mean_altitude_error_km"]
            total_accuracy  += batch_metrics["accuracy_percent"]

    avg_loss    = running_loss    / len(loader)
    avg_pos_err = total_pos_error / len(loader)
    avg_alt_err = total_alt_error / len(loader)
    avg_acc     = total_accuracy  / len(loader)

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
        group_size=CONFIG["group_size"],
        val_per_group=CONFIG["val_per_group"],
        random_seed=CONFIG["random_seed"],
        num_workers=CONFIG["num_workers"]
    )

    # 2.5 Preview do dataset antes do treino
    print("\n🖼️  Gerando preview do dataset...")
    visualize_dataset_samples(
        train_ds=train_loader.dataset,
        val_ds=val_loader.dataset,
        save_dir=CONFIG["viz_dir"]
    )

    # 3. Inicializar Modelo e Otimização
    model = LunarUNet(n_channels=1, n_classes=6).to(CONFIG["device"])
    
    # Função de Perda — distância superficial normalizada para o centro XYZ
    # + L1 normalizado para dimensões e altitude
    criterion = LunarNavigationLoss(
        w_center=1.0,   # peso da distância superficial (principal)
        w_dim=0.5,      # peso do erro de largura/altura
        w_alt=0.5,      # peso do erro de altitude
    )
    
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