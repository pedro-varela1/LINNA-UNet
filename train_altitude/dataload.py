"""
Dataloader for altitude-only training.

Loads grayscale lunar images and scalar altitude values from the dataset.
No EXR files needed — only 'img/' and 'altimeter/' folders are used.
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

ALT_MIN_KM = 0.0
ALT_MAX_KM = 120.0  # Maximum expected satellite altitude above the surface (km)

# CLAHE parameters (applied to every image before augmentation)
_CLAHE = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))


class AltitudeDataset(Dataset):
    """
    Dataset that pairs a grayscale lunar image with the satellite altitude (km).

    Each sample returns:
        image    (Tensor): shape (1, img_size, img_size), CLAHE-enhanced, values in [0, 1].
        altitude (Tensor): scalar, z-score normalized altitude ((alt_km - mean) / std).
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        group_size: int = 12,
        val_per_group: int = 2,
        random_seed: int = 42,
        img_size: int = 224,
        alt_mean: float = None,
        alt_std:  float = None,
    ):
        """
        Args:
            root_dir     : Dataset root — must contain 'img/' and 'altimeter/' subfolders.
            mode         : 'train' or 'val'.
            group_size   : Number of images per geographic region group.
            val_per_group: Images randomly assigned to validation per group.
            random_seed  : Seed for the train/val split.
            img_size     : Resize target for MobileNetV2 (default 224).
            alt_mean     : Pre-computed mean altitude (km) for z-score. If None, computed from this split.
            alt_std      : Pre-computed std  altitude (km) for z-score. If None, computed from this split.
        """
        self.mode = mode
        self.img_size = img_size

        # --- Collect and match files by name ---
        img_files = glob.glob(os.path.join(root_dir, "img", "*.png"))
        alt_files = glob.glob(os.path.join(root_dir, "altimeter", "*.txt"))

        img_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
        alt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in alt_files}

        common = sorted(img_dict.keys() & alt_dict.keys())
        self.img_files = [img_dict[name] for name in common]
        self.alt_files = [alt_dict[name] for name in common]

        # --- Group-based train/val split ---
        # Consecutive groups of `group_size` represent the same geographic region.
        # `val_per_group` images are randomly drawn for validation; the rest go to train.
        # Incomplete last groups are fully assigned to train.
        n = len(self.img_files)
        train_idx, val_idx = [], []
        rng = random.Random(random_seed)

        for start in range(0, n, group_size):
            group = list(range(start, min(start + group_size, n)))
            if len(group) == group_size:
                val_set = set(rng.sample(group, val_per_group))
                for idx in group:
                    (val_idx if idx in val_set else train_idx).append(idx)
            else:
                train_idx.extend(group)  # Incomplete group → all train

        self.indices = train_idx if mode == "train" else val_idx

        # --- Z-score stats for altitude labels ---
        # Load all altitude values for this split to compute (or receive) mean/std.
        all_alts = np.array(
            [float(open(self.alt_files[i]).read().strip()) for i in self.indices],
            dtype=np.float32,
        )
        if alt_mean is None:
            self.alt_mean = float(all_alts.mean())
            self.alt_std  = float(all_alts.std()) if all_alts.std() > 0 else 1.0
        else:
            self.alt_mean = alt_mean
            self.alt_std  = alt_std

        print(
            f"[AltitudeDataset/{mode}]  {len(train_idx)} train / {len(val_idx)} val  "
            f"(group_size={group_size}, val_per_group={val_per_group}, seed={random_seed})  "
            f"alt_mean={self.alt_mean:.2f} km  alt_std={self.alt_std:.2f} km"
        )

        # --- Augmentations (train only) ---
        #
        # Because altitude is a scalar property of the satellite orbit, it is
        # completely independent of how the image is oriented or cropped.
        # This means we can freely apply GEOMETRIC transforms (flips, rotations,
        # crops) in addition to photometric ones.
        if mode == "train":
            self.augmentations = A.Compose([
                # --- Geometric (safe because altitude is orientation-invariant) ---
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Full 360° rotation: no preferred "up" direction in orbit imagery
                A.RandomRotate90(p=0.5),
                # Random crop + resize: simulates different zoom levels / partial views
                A.RandomResizedCrop(
                    size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.4
                ),
                # Coarse dropout: forces the model not to rely on a single surface feature
                A.CoarseDropout(
                    num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3
                ),
                # --- Photometric ---
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.GaussNoise(noise_scale_factor=1.0, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ])
        else:
            self.augmentations = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]

        # Load and resize image
        image = cv2.imread(self.img_files[real_idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # CLAHE contrast enhancement (applied before augmentation)
        image = _CLAHE.apply(image)

        # Apply augmentations
        if self.augmentations is not None:
            image = self.augmentations(image=image)["image"]

        # Normalize to [0, 1] and add channel dim → (1, H, W)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Load altitude and apply z-score normalization
        with open(self.alt_files[real_idx]) as f:
            altitude_km = float(f.read().strip())
        altitude_norm = (altitude_km - self.alt_mean) / self.alt_std

        return (
            torch.from_numpy(image),
            torch.tensor(altitude_norm, dtype=torch.float32),
        )


def get_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    img_size: int = 224,
    group_size: int = 12,
    val_per_group: int = 2,
    random_seed: int = 42,
    num_workers: int = 4,
):
    """
    Create and return (train_loader, val_loader, alt_mean, alt_std).

    alt_mean / alt_std are computed from the training split and shared with
    validation so both use the same z-score normalization.
    """
    shared = dict(
        group_size=group_size,
        val_per_group=val_per_group,
        random_seed=random_seed,
        img_size=img_size,
    )
    # Train dataset computes its own stats
    train_ds = AltitudeDataset(dataset_root, mode="train", **shared)
    # Val dataset reuses training stats so normalization is consistent
    val_ds   = AltitudeDataset(
        dataset_root, mode="val",
        alt_mean=train_ds.alt_mean, alt_std=train_ds.alt_std,
        **shared,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.alt_mean, train_ds.alt_std


# --- Quick sanity check ---
if __name__ == "__main__":
    ROOT = "../../LINNA-Crater/LunarLanding_Dataset/LunarLanding_Dataset"

    if not os.path.exists(ROOT):
        print(f"Path not found: {ROOT}")
    else:
        train_loader, val_loader, alt_mean, alt_std = get_dataloaders(ROOT, batch_size=4)
        train_loader, val_loader, alt_mean, alt_std = get_dataloaders(ROOT, batch_size=4)
        images, altitudes = next(iter(train_loader))

        print("✅  Dataloader OK")
        print(f"   Image shape   : {images.shape}")           # (4, 1, 224, 224)
        print(f"   Altitude (z)  : {altitudes}")              # z-score values
        print(f"   Alt (km)      : {altitudes * alt_std + alt_mean}")
