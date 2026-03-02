"""
Training script for CraterCenterNet — single-network crater detection.

Detects all lunar craters with radius in [5, 20] km in a 512×512 image,
predicting a centre heatmap and a co-located radius map (both at 128×128).
No altimeter input required: the network infers scale implicitly from the
visual appearance of craters at different orbital altitudes.

Usage:
    python train.py
"""

import os
import csv
import time

import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataload import get_dataloaders, RADIUS_MIN_KM, RADIUS_MAX_KM, OUT_SIZE
from model  import CraterCenterNet
from loss   import CraterLoss
from test   import compute_metrics, extract_peaks, decode_radius_km

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "dataset_root":  "../../LINNA-Crater/LunarLanding_Dataset/LunarLanding_Dataset",
    "num_epochs":    60,
    "batch_size":    8,
    "learning_rate": 1e-4,
    "weight_decay":  1e-4,
    "group_size":    12,
    "val_per_group": 2,
    "random_seed":   42,
    "num_workers":   4,
    # Loss weights
    "w_hm":          1.0,
    "w_rad":         2.0,
    # Metric thresholds
    "hm_thresh":     0.3,     # peak detection threshold
    "nms_pool":      5,       # NMS kernel size (output pixels)
    "max_dist_out":  4.0,     # max centre distance for TP match (output pixels)
    # Saves
    "save_dir":      "./checkpoints",
    "csv_log_file":  "./training_metrics_craters.csv",
    "viz_dir":       "./visualizations",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_lhm = total_lrad = 0.0

    for images, gt_hm, gt_rad, regmask in tqdm(loader, desc="  Train", leave=False):
        images  = images.to(device)
        gt_hm   = gt_hm.to(device)
        gt_rad  = gt_rad.to(device)
        regmask = regmask.to(device)

        optimizer.zero_grad()
        pred_hm, pred_rad = model(images)
        loss, l_hm, l_rad = criterion(pred_hm, pred_rad, gt_hm, gt_rad, regmask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        total_lhm  += l_hm.item()
        total_lrad += l_rad.item()

    n = len(loader)
    return total_loss / n, total_lhm / n, total_lrad / n


def validate(model, loader, criterion, device, hm_thresh, nms_pool, max_dist_out):
    model.eval()
    total_loss = total_lhm = total_lrad = 0.0
    agg_tp = agg_fp = agg_fn = 0
    agg_rad_err = []

    with torch.no_grad():
        for images, gt_hm, gt_rad, regmask in tqdm(loader, desc="  Val  ", leave=False):
            images  = images.to(device)
            gt_hm   = gt_hm.to(device)
            gt_rad  = gt_rad.to(device)
            regmask = regmask.to(device)

            pred_hm, pred_rad = model(images)
            loss, l_hm, l_rad = criterion(pred_hm, pred_rad, gt_hm, gt_rad, regmask)

            total_loss += loss.item()
            total_lhm  += l_hm.item()
            total_lrad += l_rad.item()

            m = compute_metrics(pred_hm, pred_rad, gt_hm, gt_rad, regmask,
                                hm_thresh=hm_thresh, nms_pool=nms_pool,
                                max_dist_out=max_dist_out)
            agg_tp += m["tp"]
            agg_fp += m["fp"]
            agg_fn += m["fn"]
            # Collect raw errors for MAE
            # (compute_metrics returns mean; we accumulate tp-weighted sums)
            if m["tp"] > 0:
                agg_rad_err.append((m["radius_mae_km"], m["tp"]))

    n = len(loader)
    precision = agg_tp / max(agg_tp + agg_fp, 1)
    recall    = agg_tp / max(agg_tp + agg_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-6)
    # Weighted mean radius MAE
    if agg_rad_err:
        total_w   = sum(w for _, w in agg_rad_err)
        rad_mae   = sum(e * w for e, w in agg_rad_err) / total_w
    else:
        rad_mae   = 0.0

    return {
        "loss":      total_loss / n,
        "l_hm":      total_lhm  / n,
        "l_rad":     total_lrad / n,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "rad_mae_km": rad_mae,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def visualize_predictions(model, val_loader, epoch, device, save_dir,
                          hm_thresh, nms_pool, num_samples=4):
    """Save a grid of: image | GT heatmap | Pred heatmap | overlay with circles."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    images, gt_hm, gt_rad, regmask = next(iter(val_loader))
    images = images[:num_samples].to(device)

    with torch.no_grad():
        pred_hm, pred_rad = model(images)

    images   = images.cpu()
    pred_hm  = pred_hm.cpu()
    pred_rad = pred_rad.cpu()
    gt_hm    = gt_hm[:num_samples].cpu()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    col_titles = ["Image", "GT heatmap", "Pred heatmap", "Detections"]

    for i in range(num_samples):
        img_np   = images[i, 0].numpy()
        gt_np    = gt_hm[i, 0].numpy()
        pred_np  = pred_hm[i, 0].numpy()

        # Extract peaks and draw circles
        peaks = extract_peaks(pred_hm[i, 0], pred_rad[i, 0],
                              hm_thresh=hm_thresh, nms_pool=nms_pool)

        # Build overlay: upscale heatmap to image size for visual
        import cv2
        from dataload import OUT_STRIDE
        overlay = (img_np * 255).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        for cx_out, cy_out, r_norm in peaks:
            cx_img = cx_out * OUT_STRIDE
            cy_img = cy_out * OUT_STRIDE
            r_km   = decode_radius_km(r_norm)
            # Visual radius proportional to km
            r_vis  = max(3, int(r_km * 2))
            cv2.circle(overlay_bgr, (cx_img, cy_img), r_vis, (0, 255, 0), 2)
            cv2.putText(overlay_bgr, f"{r_km:.1f}", (cx_img + 4, cy_img - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        row = axes[i] if num_samples > 1 else axes
        row[0].imshow(img_np,  cmap="gray");   row[0].axis("off")
        row[1].imshow(gt_np,   cmap="hot");    row[1].axis("off")
        row[2].imshow(pred_np, cmap="hot");    row[2].axis("off")
        row[3].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)); row[3].axis("off")
        if i == 0:
            for ax, t in zip(row, col_titles):
                ax.set_title(t, fontsize=9)

    plt.suptitle(f"Epoch {epoch}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   Viz saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)
    os.makedirs(cfg["viz_dir"],  exist_ok=True)
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    # --- Data ---
    train_loader, val_loader = get_dataloaders(
        cfg["dataset_root"],
        batch_size    = cfg["batch_size"],
        group_size    = cfg["group_size"],
        val_per_group = cfg["val_per_group"],
        random_seed   = cfg["random_seed"],
        num_workers   = cfg["num_workers"],
    )

    # --- Model ---
    model     = CraterCenterNet(pretrained=True).to(device)
    criterion = CraterLoss(w_hm=cfg["w_hm"], w_rad=cfg["w_rad"])
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6
    )

    # --- CSV log ---
    with open(cfg["csv_log_file"], "w", newline="") as f:
        csv.writer(f).writerow([
            "Epoch",
            "Train_Loss", "Train_L_HM", "Train_L_Rad",
            "Val_Loss",   "Val_L_HM",   "Val_L_Rad",
            "Val_Precision", "Val_Recall", "Val_F1", "Val_Rad_MAE_km",
            "Time_sec",
        ])

    best_val_loss = float("inf")
    best_f1       = 0.0

    for epoch in range(1, cfg["num_epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{cfg['num_epochs']} ---")
        t0 = time.time()

        # Train
        tr_loss, tr_lhm, tr_lrad = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_m = validate(
            model, val_loader, criterion, device,
            hm_thresh    = cfg["hm_thresh"],
            nms_pool     = cfg["nms_pool"],
            max_dist_out = cfg["max_dist_out"],
        )

        elapsed = time.time() - t0
        scheduler.step(val_m["loss"])

        # Print summary
        print(f"  Train  loss={tr_loss:.4f}  l_hm={tr_lhm:.4f}  l_rad={tr_lrad:.4f}")
        print(f"  Val    loss={val_m['loss']:.4f}  l_hm={val_m['l_hm']:.4f}  "
              f"l_rad={val_m['l_rad']:.4f}")
        print(f"  Det    P={val_m['precision']:.3f}  R={val_m['recall']:.3f}  "
              f"F1={val_m['f1']:.3f}  RadMAE={val_m['rad_mae_km']:.2f} km")
        print(f"  Time   {elapsed:.1f}s")

        # CSV
        with open(cfg["csv_log_file"], "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{tr_loss:.6f}",  f"{tr_lhm:.6f}",  f"{tr_lrad:.6f}",
                f"{val_m['loss']:.6f}", f"{val_m['l_hm']:.6f}", f"{val_m['l_rad']:.6f}",
                f"{val_m['precision']:.4f}", f"{val_m['recall']:.4f}",
                f"{val_m['f1']:.4f}", f"{val_m['rad_mae_km']:.4f}",
                f"{elapsed:.1f}",
            ])

        # Save checkpoints
        state = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_loss":   val_m["loss"],
            "val_f1":     val_m["f1"],
        }

        torch.save(state, os.path.join(cfg["save_dir"], "crater_last_model.pth"))

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save(state, os.path.join(cfg["save_dir"], "crater_best_loss_model.pth"))
            print("  ★  New best validation loss — saved.")

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(state, os.path.join(cfg["save_dir"], "crater_best_f1_model.pth"))
            print(f"  ★  New best F1={best_f1:.4f} — saved.")

        # Visualise every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            visualize_predictions(
                model, val_loader, epoch, device,
                save_dir=cfg["viz_dir"],
                hm_thresh=cfg["hm_thresh"],
                nms_pool=cfg["nms_pool"],
            )

    print("\nTraining complete!")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Best val F1   : {best_f1:.4f}")


if __name__ == "__main__":
    main()
