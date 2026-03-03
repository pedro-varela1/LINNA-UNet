"""
Prediction visualization for the altitude model.

Layout (per row):
    [Original Image] ---> [CLAHE + Resized] ---> [ Result Block ]

Three examples are shown (3 rows).

Usage:
    Set CHECKPOINT_PATH and DATASET_ROOT, then run:
        python plot_predictions.py
"""

import os
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

from dataload import AltitudeDataset
from model import AltitudeMobileNet

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "./checkpoints/best_model.pth"   # <-- set your checkpoint here
DATASET_ROOT    = "../../LINNA-Crater/LunarLanding_Dataset/LunarLanding_Dataset"
IMG_SIZE        = 224
RANDOM_SEED     = 7       # change to pick different samples
N_EXAMPLES      = 3
TOLERANCE_KM     = 5.0     # for coloring the error in the result block
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------

_CLAHE = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))


def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load model and z-score stats from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = AltitudeMobileNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    alt_mean = float(ckpt["alt_mean"])
    alt_std  = float(ckpt["alt_std"])
    return model, alt_mean, alt_std


def preprocess(img_bgr: np.ndarray) -> tuple:
    """
    Returns:
        orig_rgb   : original image as RGB uint8 for display.
        clahe_rgb  : CLAHE-enhanced + resized image as RGB uint8 for display.
        tensor     : (1, 1, H, W) float32 tensor ready for inference.
    """
    orig_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orig_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    resized   = cv2.resize(orig_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    clahe_img = _CLAHE.apply(resized)
    clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

    tensor = torch.from_numpy(
        clahe_img.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

    return orig_rgb, clahe_rgb, tensor


def pick_samples(dataset_root: str, n: int, seed: int) -> list:
    """
    Returns a list of dicts with keys: img_path, alt_km.
    Samples are picked from the validation split to avoid training bias.
    """
    import glob

    img_files = sorted(glob.glob(os.path.join(dataset_root, "img", "*.png")))
    alt_files = sorted(glob.glob(os.path.join(dataset_root, "altimeter", "*.txt")))

    img_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
    alt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in alt_files}
    common   = sorted(img_dict.keys() & alt_dict.keys())

    rng = random.Random(seed)
    chosen = rng.sample(common, n)

    return [
        {
            "img_path": img_dict[name],
            "alt_km":   float(open(alt_dict[name]).read().strip()),
        }
        for name in chosen
    ]


def draw_arrow(fig, ax_src, ax_dst):
    """Draw a curved arrow between two axes using figure coordinates."""
    # Get bounding boxes in figure fraction coordinates
    src_bbox = ax_src.get_position()
    dst_bbox = ax_dst.get_position()

    x_start = src_bbox.x1          # right edge of source
    y_start = src_bbox.y0 + src_bbox.height / 2.0
    x_end   = dst_bbox.x0          # left edge of destination
    y_end   = dst_bbox.y0 + dst_bbox.height / 2.0

    arrow = FancyArrowPatch(
        posA=(x_start, y_start),
        posB=(x_end,   y_end),
        transform=fig.transFigure,
        arrowstyle="-|>",
        color="#555555",
        lw=1.8,
        mutation_scale=14,
        connectionstyle="arc3,rad=0.0",
        zorder=10,
        alpha=0.75,
    )
    fig.add_artist(arrow)


def result_text(ax, real_km: float, pred_km: float):
    """Fill an axis with a styled text result block."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    error_km = abs(pred_km - real_km)
    color_err = "#2ECC71" if error_km <= TOLERANCE_KM else "#E74C3C"

    # Background box
    ax.add_patch(plt.Rectangle(
        (0.05, 0.05), 0.90, 0.90,
        transform=ax.transAxes,
        facecolor="#1C2833",
        edgecolor="#5D6D7E",
        linewidth=1.5,
        zorder=1,
        clip_on=False,
    ))

    ax.text(0.50, 0.82, "Altitude Prediction",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#AAB7B8", transform=ax.transAxes, zorder=2)

    ax.text(0.50, 0.62, f"Real:  {real_km:.1f} km",
            ha="center", va="center", fontsize=11, color="white",
            transform=ax.transAxes, zorder=2)

    ax.text(0.50, 0.45, f"Pred:  {pred_km:.1f} km",
            ha="center", va="center", fontsize=11, color="#5DADE2",
            transform=ax.transAxes, zorder=2)

    ax.text(0.50, 0.26, f"Error: {error_km:.1f} km",
            ha="center", va="center", fontsize=11, color=color_err,
            fontweight="bold", transform=ax.transAxes, zorder=2)

    threshold_label = "✓  within " f"{TOLERANCE_KM} km" if error_km <= TOLERANCE_KM else "✗  above " f"{TOLERANCE_KM} km"
    ax.text(0.50, 0.11, threshold_label,
            ha="center", va="center", fontsize=8.5, color=color_err,
            transform=ax.transAxes, zorder=2)


def main():
    # --- Load model ---
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model, alt_mean, alt_std = load_model(CHECKPOINT_PATH, DEVICE)
    print(f"  alt_mean={alt_mean:.2f} km  alt_std={alt_std:.2f} km  device={DEVICE}")

    # --- Pick samples ---
    samples = pick_samples(DATASET_ROOT, N_EXAMPLES, RANDOM_SEED)

    # --- Figure layout ---
    # 3 rows x 5 cols: [orig] [gap] [clahe] [gap] [result]
    # Gaps are narrow axes used only for arrows
    col_widths  = [3.0, 0.6, 3.0, 0.6, 2.2]
    fig, axes = plt.subplots(
        N_EXAMPLES, 5,
        figsize=(sum(col_widths) * 1.05, N_EXAMPLES * 3.4),
        gridspec_kw={"width_ratios": col_widths, "wspace": 0.08, "hspace": 0.35},
    )

    fig.patch.set_facecolor("#0D1117")

    col_labels = ["Original Image", "", "CLAHE + Resized  (224×224)", "", "Model Output"]

    for row, sample in enumerate(samples):
        img_bgr = cv2.imread(sample["img_path"])
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {sample['img_path']}")

        orig_rgb, clahe_rgb, tensor = preprocess(img_bgr)

        # --- Inference ---
        with torch.no_grad():
            pred_norm = model(tensor.to(DEVICE))
        pred_km = float(pred_norm.item() * alt_std + alt_mean)
        real_km = sample["alt_km"]

        # --- Column 0: Original ---
        ax_orig = axes[row][0]
        ax_orig.imshow(orig_rgb, cmap="gray")
        ax_orig.axis("off")
        ax_orig.set_facecolor("#0D1117")
        if row == 0:
            ax_orig.set_title(col_labels[0], color="white", fontsize=9, pad=5)

        # --- Column 1: Arrow gap ---
        ax_gap1 = axes[row][1]
        ax_gap1.axis("off")
        ax_gap1.set_facecolor("#0D1117")

        # --- Column 2: CLAHE ---
        ax_clahe = axes[row][2]
        ax_clahe.imshow(clahe_rgb, cmap="gray")
        ax_clahe.axis("off")
        ax_clahe.set_facecolor("#0D1117")
        if row == 0:
            ax_clahe.set_title(col_labels[2], color="white", fontsize=9, pad=5)

        # --- Column 3: Arrow gap ---
        ax_gap2 = axes[row][3]
        ax_gap2.axis("off")
        ax_gap2.set_facecolor("#0D1117")

        # --- Column 4: Result block ---
        ax_res = axes[row][4]
        result_text(ax_res, real_km, pred_km)
        ax_res.set_facecolor("#0D1117")
        if row == 0:
            ax_res.set_title(col_labels[4], color="white", fontsize=9, pad=5)

    fig.suptitle(
        "Altitude Estimation — Prediction Examples",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )

    # --- Draw arrows after layout is finalized ---
    fig.canvas.draw()  # force layout computation
    for row in range(N_EXAMPLES):
        draw_arrow(fig, axes[row][0], axes[row][2])   # orig  → clahe
        draw_arrow(fig, axes[row][2], axes[row][4])   # clahe → result

    plt.savefig("prediction_examples.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("Saved: prediction_examples.png")


if __name__ == "__main__":
    main()
