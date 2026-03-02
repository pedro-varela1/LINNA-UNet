"""
Training script — CraterShuffleNet  (5-20 km craters, direct regression).

ShuffleNetV2 x1.0 backbone → GAP → FC(256) → N_MAX slots of (cx, cy, r) + conf.
~2.3 M parameters.  Typical epoch time: ~1-2 min on a 6 GB GPU.

Usage:
    python train.py
"""

import os, csv, time
import cv2, numpy as np
import torch, torch.optim as optim
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataload import get_dataloaders, IMG_SIZE, N_MAX
from model   import CraterShuffleNet
from loss    import CraterLoss
from test    import compute_metrics, decode_predictions

CONFIG = {
    "dataset_root":  "../../LINNA-Crater/LunarLanding_Dataset/LunarLanding_Dataset",
    "num_epochs":    80,
    "batch_size":    16,         # ShuffleNet is light — 16 fits easily
    "learning_rate": 3e-4,
    "weight_decay":  1e-4,
    "group_size":    12,
    "val_per_group": 2,
    "random_seed":   42,
    "num_workers":   4,
    # Loss
    "w_reg":         2.0,
    "w_conf":        1.0,
    # Decode / evaluate
    "conf_thresh":     0.5,
    "nms_iou_thresh":  0.3,
    "match_iou_thresh":0.3,
    # IO
    "save_dir": "./checkpoints",
    "csv_log":  "./training_metrics.csv",
    "viz_dir":  "./visualizations",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ── Training step ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    tot_loss = tot_conf = tot_reg = 0.0

    for images, gt_boxes, mask in tqdm(loader, desc="  Train", leave=False):
        images   = images.to(device)
        gt_boxes = gt_boxes.to(device)
        mask     = mask.to(device)

        preds, logits = model(images)
        loss, lc, lr  = loss_fn(preds, logits, gt_boxes, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        tot_loss += loss.item()
        tot_conf += lc.item()
        tot_reg  += lr.item()

    n = len(loader)
    return tot_loss/n, tot_conf/n, tot_reg/n


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    all_preds = all_logits = all_gt = all_mask = None

    for images, gt_boxes, mask in tqdm(loader, desc="  Val  ", leave=False):
        images   = images.to(device)
        preds, logits = model(images)
        # Accumulate on CPU
        p  = preds.cpu();  l = logits.cpu()
        g  = gt_boxes.cpu(); m = mask.cpu()
        all_preds  = p  if all_preds  is None else torch.cat([all_preds,  p])
        all_logits = l  if all_logits is None else torch.cat([all_logits, l])
        all_gt     = g  if all_gt     is None else torch.cat([all_gt,     g])
        all_mask   = m  if all_mask   is None else torch.cat([all_mask,   m])

    return compute_metrics(all_preds, all_logits, all_gt, all_mask,
                           conf_thresh=cfg["conf_thresh"],
                           nms_iou_thresh=cfg["nms_iou_thresh"],
                           match_iou_thresh=cfg["match_iou_thresh"])


# ── Visualisation ─────────────────────────────────────────────────────────────
@torch.no_grad()
def visualize(model, val_loader, epoch, device, cfg, num_samples=4):
    model.eval()
    os.makedirs(cfg["viz_dir"], exist_ok=True)

    images, gt_boxes, mask = next(iter(val_loader))
    B = min(num_samples, images.shape[0])
    preds, logits = model(images[:B].to(device))
    preds  = preds.cpu(); logits = logits.cpu()

    fig, axes = plt.subplots(B, 2, figsize=(10, 5*B))
    if B == 1: axes = [axes]

    for i in range(B):
        img_np = (images[i, 0].numpy() * 255).astype(np.uint8)

        def draw(circles, color):
            out = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            for cx_n, cy_n, r_n, *rest in circles:
                cx = int(cx_n * IMG_SIZE)
                cy = int(cy_n * IMG_SIZE)
                r  = max(3, int(r_n * IMG_SIZE))
                cv2.circle(out, (cx, cy), r, color, 2)
                cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
            return out

        # GT circles
        n_gt = int(mask[i].sum())
        gt_circles = [(float(gt_boxes[i,j,0]), float(gt_boxes[i,j,1]),
                       float(gt_boxes[i,j,2]), 1.0) for j in range(n_gt)]
        gt_img = draw(gt_circles, (0, 255, 0))

        # Predicted circles
        pred_dets = decode_predictions(preds[i], logits[i],
                                       cfg["conf_thresh"], cfg["nms_iou_thresh"])
        pred_img = draw(pred_dets, (0, 200, 255))

        row = axes[i]
        row[0].imshow(cv2.cvtColor(gt_img,   cv2.COLOR_BGR2RGB))
        row[0].set_title(f"GT ({n_gt} craters)"); row[0].axis("off")
        row[1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
        row[1].set_title(f"Pred ({len(pred_dets)} dets, thresh={cfg['conf_thresh']:.1f})")
        row[1].axis("off")

    plt.suptitle(f"Epoch {epoch} — green=GT  yellow=Pred  red dot=centre", fontsize=9)
    plt.tight_layout()
    path = os.path.join(cfg["viz_dir"], f"epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   Viz → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    train_loader, val_loader = get_dataloaders(
        cfg["dataset_root"], batch_size=cfg["batch_size"],
        group_size=cfg["group_size"], val_per_group=cfg["val_per_group"],
        random_seed=cfg["random_seed"], num_workers=cfg["num_workers"])

    model   = CraterShuffleNet(pretrained=True).to(device)
    loss_fn = CraterLoss(w_reg=cfg["w_reg"], w_conf=cfg["w_conf"])
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8)

    print(f"Parameters: {model.count_parameters():,}")

    with open(cfg["csv_log"], "w", newline="") as f:
        csv.writer(f).writerow([
            "Epoch", "Train_Loss", "Train_Conf", "Train_Reg",
            "Val_P", "Val_R", "Val_F1", "Val_RadMAE_norm", "Time_s"])

    best_f1 = 0.0

    for epoch in range(1, cfg["num_epochs"] + 1):
        print(f"--- Epoch {epoch}/{cfg['num_epochs']} ---")
        t0 = time.time()

        tr_loss, tr_conf, tr_reg = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        val_m = validate(model, val_loader, device, cfg)
        elapsed = time.time() - t0
        scheduler.step(val_m["f1"])

        print(f"  Train  loss={tr_loss:.4f}  conf={tr_conf:.4f}  reg={tr_reg:.4f}")
        print(f"  Val    P={val_m['precision']:.3f}  R={val_m['recall']:.3f}  "
              f"F1={val_m['f1']:.3f}  RadMAE={val_m['radius_mae_norm']:.4f}  "
              f"({elapsed:.1f}s)")

        with open(cfg["csv_log"], "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{tr_loss:.6f}", f"{tr_conf:.6f}", f"{tr_reg:.6f}",
                f"{val_m['precision']:.4f}", f"{val_m['recall']:.4f}",
                f"{val_m['f1']:.4f}", f"{val_m['radius_mae_norm']:.6f}",
                f"{elapsed:.1f}"])

        state = {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "val_f1": val_m["f1"]}
        torch.save(state, os.path.join(cfg["save_dir"], "last_model.pth"))

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(state, os.path.join(cfg["save_dir"], "best_f1_model.pth"))
            print(f"  ★  Best F1={best_f1:.4f} — saved.")

        if epoch % 5 == 0 or epoch == 1:
            visualize(model, val_loader, epoch, device, cfg)

    print(f"Done.  Best F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
