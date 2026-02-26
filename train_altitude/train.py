"""
Training script for altitude-only prediction using ResNet18.

Usage:
    python train.py
"""

import os
import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataload import get_dataloaders
from model import AltitudeResNet
from test import compute_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "dataset_root":  "/media/ita/DATA/LINNA-Crater/LunarLanding_Dataset/",
    "num_epochs":    50,
    "batch_size":    8,
    "learning_rate": 1e-4,
    "img_size":      224,
    "group_size":    12,
    "val_per_group": 2,
    "random_seed":   42,
    "threshold_km":  5.0,       # Accuracy threshold: error < 5 km counts as correct
    "save_dir":      "./checkpoints",
    "csv_log_file":  "./training_metrics_altitude.csv",
    "num_workers":   4,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
}
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device, threshold_km):
    model.train()
    total_loss, total_mae, total_acc = 0.0, 0.0, 0.0

    for images, targets in tqdm(loader, desc="  Train", unit="batch", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            m = compute_metrics(preds, targets, threshold_km=threshold_km)
            total_mae += m["mae_km"]
            total_acc += m["accuracy_percent"]

    n = len(loader)
    return total_loss / n, total_mae / n, total_acc / n


def validate(model, loader, criterion, device, threshold_km):
    model.eval()
    total_loss, total_mae, total_acc = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="  Val  ", unit="batch", leave=False):
            images, targets = images.to(device), targets.to(device)

            preds = model(images)
            loss  = criterion(preds, targets)

            total_loss += loss.item()
            m = compute_metrics(preds, targets, threshold_km=threshold_km)
            total_mae += m["mae_km"]
            total_acc += m["accuracy_percent"]

    n = len(loader)
    return total_loss / n, total_mae / n, total_acc / n


def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # Initialize CSV log
    with open(CONFIG["csv_log_file"], "w", newline="") as f:
        csv.writer(f).writerow([
            "Epoch",
            "Train_Loss", "Train_MAE_km", "Train_Acc_%",
            "Val_Loss",   "Val_MAE_km",   "Val_Acc_%",
            "Time_sec",
        ])

    print(f"🚀  Device: {CONFIG['device']}")

    # Data
    train_loader, val_loader = get_dataloaders(
        CONFIG["dataset_root"],
        batch_size    = CONFIG["batch_size"],
        img_size      = CONFIG["img_size"],
        group_size    = CONFIG["group_size"],
        val_per_group = CONFIG["val_per_group"],
        random_seed   = CONFIG["random_seed"],
        num_workers   = CONFIG["num_workers"],
    )

    # Model, loss, optimizer
    model     = AltitudeResNet().to(CONFIG["device"])
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{CONFIG['num_epochs']} ---")

        tr_loss, tr_mae, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            CONFIG["device"], CONFIG["threshold_km"]
        )
        val_loss, val_mae, val_acc = validate(
            model, val_loader, criterion,
            CONFIG["device"], CONFIG["threshold_km"]
        )

        elapsed = time.time() - t0
        scheduler.step(val_loss)

        # Log to CSV
        with open(CONFIG["csv_log_file"], "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{tr_loss:.4f}",  f"{tr_mae:.2f}",  f"{tr_acc:.2f}",
                f"{val_loss:.4f}", f"{val_mae:.2f}",  f"{val_acc:.2f}",
                f"{elapsed:.1f}",
            ])

        print(f"   Train — Loss: {tr_loss:.4f}  MAE: {tr_mae:.2f} km  Acc: {tr_acc:.1f}%")
        print(f"   Val   — Loss: {val_loss:.4f}  MAE: {val_mae:.2f} km  Acc: {val_acc:.1f}%")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CONFIG["save_dir"], "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"   ⭐  Best model saved → {best_path}")

        # Always save latest checkpoint
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "last_model.pth"))

    print("\n🏁  Training complete!")


if __name__ == "__main__":
    main()
