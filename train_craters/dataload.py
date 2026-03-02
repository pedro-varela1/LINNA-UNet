"""
CenterNet-style dataloader for lunar crater detection.

Each sample returns:
  image   : Tensor (1, IMG_SIZE, IMG_SIZE)   — grayscale, normalised [0,1]
  heatmap : Tensor (1, OUT_H, OUT_W)         — Gaussian blobs at crater centres (CenterNet style)
  radmap  : Tensor (1, OUT_H, OUT_W)         — normalised radius at every crater peak pixel
  regmask : Tensor (1, OUT_H, OUT_W)         — binary: 1 exactly at crater centre pixels

Only craters whose radius falls in [RADIUS_MIN_KM, RADIUS_MAX_KM] are used.
Radius is expressed in km (converted from normalised-pixel labels via the EXR footprint).
Normalised radius = (r_km - RADIUS_MIN_KM) / (RADIUS_MAX_KM - RADIUS_MIN_KM)  →  [0, 1]

Dataset folder layout expected:
  <root>/img/                  *.png
  <root>/lat_lon_exr/          *.exr   (R = lat deg, G = lon deg)
  <root>/YOLO1_centroid_labels/*.txt   (class cx cy r_normalised)
  <root>/altimeter/            *.txt   (altitude km — not used for targets, only metadata)
"""

import os
import glob
import random
import math

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import OpenEXR
import Imath

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOON_RADIUS_KM = 1737.4
IMG_SIZE       = 512       # native image resolution
OUT_STRIDE     = 4         # output heatmap is 1/4 of input → 128×128
OUT_SIZE       = IMG_SIZE // OUT_STRIDE   # 128

RADIUS_MIN_KM  = 5.0
RADIUS_MAX_KM  = 20.0

# Gaussian sigma for heatmap rendering (in output-map pixels).
# We use the object-size-adaptive formula from CenterNet:
#   sigma = max(1, r_out_pixels / 3)
# where r_out_pixels = r_km / km_per_pixel_out
HEATMAP_MIN_SIGMA = 2.0   # floor so very small craters still leave a visible peak


# ---------------------------------------------------------------------------
# Helper: EXR → footprint in km
# ---------------------------------------------------------------------------
def _load_exr_latlon(exr_path: str):
    """Returns (lat_map, lon_map) as float32 arrays of shape (H, W)."""
    f = OpenEXR.InputFile(exr_path)
    dw = f.header()["dataWindow"]
    W  = dw.max.x - dw.min.x + 1
    H  = dw.max.y - dw.min.y + 1
    FT = Imath.PixelType(Imath.PixelType.FLOAT)
    lat = np.frombuffer(f.channel("R", FT), dtype=np.float32).reshape(H, W)
    lon = np.frombuffer(f.channel("G", FT), dtype=np.float32).reshape(H, W)
    return lat, lon


def _latlon_to_cartesian(lat_deg: float, lon_deg: float) -> np.ndarray:
    lr = math.radians(lat_deg)
    lo = math.radians(lon_deg)
    r  = MOON_RADIUS_KM
    return np.array([r * math.cos(lr) * math.cos(lo),
                     r * math.cos(lr) * math.sin(lo),
                     r * math.sin(lr)])


def _footprint_km(lat_map: np.ndarray, lon_map: np.ndarray):
    """Return (width_km, height_km) of the image footprint on the lunar surface."""
    H, W = lat_map.shape
    tl = _latlon_to_cartesian(lat_map[0,    0  ], lon_map[0,    0  ])
    tr = _latlon_to_cartesian(lat_map[0,    W-1], lon_map[0,    W-1])
    bl = _latlon_to_cartesian(lat_map[H-1,  0  ], lon_map[H-1,  0  ])
    w = float(np.linalg.norm(tr - tl))
    h = float(np.linalg.norm(bl - tl))
    return w, h


# ---------------------------------------------------------------------------
# Helper: Gaussian heatmap rendering
# ---------------------------------------------------------------------------
def _draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, sigma: float):
    """Render a 2-D Gaussian (Gaussian-splat) centred at (cx, cy) into heatmap in-place."""
    radius = int(3 * sigma)
    x0, x1 = max(0, cx - radius), min(heatmap.shape[1], cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(heatmap.shape[0], cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return
    xs = np.arange(x0, x1) - cx
    ys = np.arange(y0, y1) - cy
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], g)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CraterDataset(Dataset):
    """
    Pairs each lunar image (grayscale PNG) with:
      - EXR for geo-metric footprint → km-radius conversion
      - YOLO1_centroid label files   → crater centres + normalised radius

    Returns:
      image   (1, IMG_SIZE, IMG_SIZE)   float32 [0,1]
      heatmap (1, OUT_SIZE, OUT_SIZE)   float32 [0,1]   Gaussian peaks at centres
      radmap  (1, OUT_SIZE, OUT_SIZE)   float32 [0,1]   norm. radius at crater peaks
      regmask (1, OUT_SIZE, OUT_SIZE)   float32 {0,1}   1 exactly at peak pixels
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        group_size: int = 12,
        val_per_group: int = 2,
        random_seed: int = 42,
    ):
        self.root_dir = root_dir
        self.mode     = mode

        # --- Collect matching triplets: img / exr / label ---
        img_dict = {os.path.splitext(os.path.basename(p))[0]: p
                    for p in glob.glob(os.path.join(root_dir, "img", "*.png"))}
        exr_dict = {os.path.splitext(os.path.basename(p))[0]: p
                    for p in glob.glob(os.path.join(root_dir, "lat_lon_exr", "*.exr"))}
        lbl_dict = {os.path.splitext(os.path.basename(p))[0]: p
                    for p in glob.glob(os.path.join(root_dir, "YOLO1_centroid_labels", "*.txt"))}

        common = sorted(img_dict.keys() & exr_dict.keys() & lbl_dict.keys())
        self.img_files = [img_dict[n] for n in common]
        self.exr_files = [exr_dict[n] for n in common]
        self.lbl_files = [lbl_dict[n] for n in common]

        # --- Group-based train / val split (same strategy as train_altitude) ---
        n = len(self.img_files)
        train_idx, val_idx = [], []
        rng = random.Random(random_seed)
        for start in range(0, n, group_size):
            group = list(range(start, min(start + group_size, n)))
            if len(group) == group_size:
                val_set = set(rng.sample(group, val_per_group))
                for i in group:
                    (val_idx if i in val_set else train_idx).append(i)
            else:
                train_idx.extend(group)

        self.indices = train_idx if mode == "train" else val_idx
        print(f"[CraterDataset/{mode}]  {len(train_idx)} train / {len(val_idx)} val  "
              f"(seed={random_seed})")

        # --- Augmentations (train only) ---
        # We use albumentations with keypoint support so crater centres are
        # transformed consistently with the image.
        kp_params = A.KeypointParams(format="xy", remove_invisible=True)

        if mode == "train":
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.GaussNoise(noise_scale_factor=1.0, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ], keypoint_params=kp_params)
        else:
            self.aug = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]

        # --- 1. Load image ---
        image = cv2.imread(self.img_files[real_idx], cv2.IMREAD_GRAYSCALE)
        if image.shape != (IMG_SIZE, IMG_SIZE):
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        # --- 2. Load EXR footprint → km-per-pixel ---
        lat_map, lon_map = _load_exr_latlon(self.exr_files[real_idx])
        w_km, h_km = _footprint_km(lat_map, lon_map)
        km_per_px_x = w_km / IMG_SIZE
        km_per_px_y = h_km / IMG_SIZE
        # km per output-map pixel
        km_per_out_x = w_km / OUT_SIZE
        km_per_out_y = h_km / OUT_SIZE

        # --- 3. Load labels, convert to km, filter 5-20 km ---
        craters = []   # list of (cx_img, cy_img, r_km)
        with open(self.lbl_files[real_idx]) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                _, cx_n, cy_n, r_n = map(float, parts)
                # r_n is normalised radius as fraction of image size
                r_km = r_n * (w_km + h_km) / 2.0
                if RADIUS_MIN_KM <= r_km <= RADIUS_MAX_KM:
                    cx_img = cx_n * IMG_SIZE
                    cy_img = cy_n * IMG_SIZE
                    craters.append((cx_img, cy_img, r_km))

        # --- 4. Apply augmentations ---
        if self.aug is not None and len(craters) > 0:
            kps = [(cx, cy) for cx, cy, _ in craters]
            r_list = [r for _, _, r in craters]
            out = self.aug(image=image, keypoints=kps)
            image = out["image"]
            aug_kps = out["keypoints"]
            # Rebuild craters with surviving keypoints (remove_invisible drops OOB)
            craters = [(kp[0], kp[1], r_list[i])
                       for i, kp in enumerate(aug_kps)]
        elif self.aug is not None:
            out = self.aug(image=image, keypoints=[])
            image = out["image"]

        # --- 5. Build heatmap, radius map, regression mask (on output grid 128×128) ---
        heatmap = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)
        radmap  = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)
        regmask = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.float32)

        for (cx_img, cy_img, r_km) in craters:
            # Map from image pixels → output-map pixels
            cx_out = int(cx_img / OUT_STRIDE)
            cy_out = int(cy_img / OUT_STRIDE)

            # Clip to [0, OUT_SIZE-1]
            if not (0 <= cx_out < OUT_SIZE and 0 <= cy_out < OUT_SIZE):
                continue

            # Sigma scaled by crater radius in output-map pixels
            r_out = r_km / ((km_per_out_x + km_per_out_y) / 2.0)
            sigma = max(HEATMAP_MIN_SIGMA, r_out / 3.0)

            _draw_gaussian(heatmap, cx_out, cy_out, sigma)

            # Radius: normalised to [0, 1]
            r_norm = (r_km - RADIUS_MIN_KM) / (RADIUS_MAX_KM - RADIUS_MIN_KM)
            r_norm = float(np.clip(r_norm, 0.0, 1.0))

            # Only overwrite radius/mask at the exact peak pixel
            radmap [cy_out, cx_out] = r_norm
            regmask[cy_out, cx_out] = 1.0

        # --- 6. Normalise image and convert to tensors ---
        image_t   = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0)
        radmap_t  = torch.from_numpy(radmap).unsqueeze(0)
        regmask_t = torch.from_numpy(regmask).unsqueeze(0)

        return image_t, heatmap_t, radmap_t, regmask_t


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def get_dataloaders(
    dataset_root: str,
    batch_size: int    = 8,
    group_size: int    = 12,
    val_per_group: int = 2,
    random_seed: int   = 42,
    num_workers: int   = 4,
):
    """Return (train_loader, val_loader)."""
    shared = dict(group_size=group_size, val_per_group=val_per_group, random_seed=random_seed)
    train_ds = CraterDataset(dataset_root, mode="train", **shared)
    val_ds   = CraterDataset(dataset_root, mode="val",   **shared)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT = "../../LINNA-Crater/LunarLanding_Dataset/LunarLanding_Dataset"

    ds = CraterDataset(ROOT, mode="train", group_size=12, val_per_group=2)
    img, hm, rm, mask = ds[0]

    print(f"image   : {img.shape}   min={img.min():.3f} max={img.max():.3f}")
    print(f"heatmap : {hm.shape}    min={hm.min():.3f} max={hm.max():.3f}")
    print(f"radmap  : {rm.shape}    min={rm.min():.3f} max={rm.max():.3f}")
    print(f"regmask : {mask.shape}  sum={mask.sum():.0f}  (num craters in target)")
    print(f"num craters (5-20km): {int(mask.sum())}")
