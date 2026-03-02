"""
Dataloader for ShuffleNet crater detection (5-20 km, filtered via EXR).

Each sample returns:
    image   (1, 512, 512)  float32  [0, 1]
    boxes   (N_MAX, 3)     float32  [cx_norm, cy_norm, r_norm]  sorted by r desc
    mask    (N_MAX,)       bool     True = valid slot
"""

import os, glob, random, math
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import OpenEXR, Imath

IMG_SIZE       = 512
N_MAX          = 16          # covers p99 of crater count (mean≈2.4, max=44)
RADIUS_MIN_KM  = 5.0
RADIUS_MAX_KM  = 20.0
MOON_RADIUS_KM = 1737.4


def _latlon_cart(lat_d, lon_d):
    lr, lo = math.radians(lat_d), math.radians(lon_d)
    R = MOON_RADIUS_KM
    return np.array([R*math.cos(lr)*math.cos(lo),
                     R*math.cos(lr)*math.sin(lo),
                     R*math.sin(lr)])

def _footprint_km(lat_map, lon_map):
    H, W = lat_map.shape
    tl = _latlon_cart(float(lat_map[0,   0  ]), float(lon_map[0,   0  ]))
    tr = _latlon_cart(float(lat_map[0,   W-1]), float(lon_map[0,   W-1]))
    bl = _latlon_cart(float(lat_map[H-1, 0  ]), float(lon_map[H-1, 0  ]))
    return float(np.linalg.norm(tr - tl)), float(np.linalg.norm(bl - tl))

def _load_exr(path):
    f  = OpenEXR.InputFile(path)
    dw = f.header()["dataWindow"]
    W  = dw.max.x - dw.min.x + 1
    H  = dw.max.y - dw.min.y + 1
    FT = Imath.PixelType(Imath.PixelType.FLOAT)
    lat = np.frombuffer(f.channel("R", FT), dtype=np.float32).reshape(H, W)
    lon = np.frombuffer(f.channel("G", FT), dtype=np.float32).reshape(H, W)
    return lat, lon


class CraterCenterDataset(Dataset):
    def __init__(self, root_dir, mode="train",
                 group_size=12, val_per_group=2, random_seed=42):
        self.mode = mode

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
        print(f"[CraterCenterDataset/{mode}]  "
              f"{len(train_idx)} train / {len(val_idx)} val")

        aug_kwargs = dict(
            bbox_params=A.BboxParams(format="yolo",
                                     label_fields=["labels"],
                                     min_area=1.0,
                                     min_visibility=0.3))
        if mode == "train":
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
                A.GaussNoise(noise_scale_factor=1.0, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            ], **aug_kwargs)
        else:
            self.aug = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        image = cv2.imread(self.img_files[real_idx], cv2.IMREAD_GRAYSCALE)
        if image.shape != (IMG_SIZE, IMG_SIZE):
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        lat_map, lon_map = _load_exr(self.exr_files[real_idx])
        w_km, h_km = _footprint_km(lat_map, lon_map)

        bboxes, labels = [], []
        with open(self.lbl_files[real_idx]) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 4:
                    continue
                _, cx_n, cy_n, r_n = map(float, parts)
                r_km = r_n * (w_km + h_km) / 2.0
                if not (RADIUS_MIN_KM <= r_km <= RADIUS_MAX_KM):
                    continue
                # Albumentations YOLO format: [cx_norm, cy_norm, w_norm, h_norm]
                # We use square bbox so w = h = 2*r
                bw = min(2 * r_n, 1.0)
                cx_c = float(np.clip(cx_n, bw/2, 1 - bw/2))
                cy_c = float(np.clip(cy_n, bw/2, 1 - bw/2))
                bboxes.append([cx_c, cy_c, bw, bw])
                labels.append(1)

        if self.aug is not None:
            out    = self.aug(image=image,
                              bboxes=bboxes if bboxes else [],
                              labels=labels if labels else [])
            image  = out["image"]
            bboxes = list(out["bboxes"])

        # Convert back to (cx_norm, cy_norm, r_norm)
        craters = [(b[0], b[1], b[2] / 2.0) for b in bboxes]
        # Sort by radius descending so the model sees largest first
        craters.sort(key=lambda x: -x[2])

        boxes = np.zeros((N_MAX, 3), dtype=np.float32)
        mask  = np.zeros((N_MAX,),   dtype=bool)
        for i, (cx, cy, r) in enumerate(craters[:N_MAX]):
            boxes[i] = [cx, cy, r]
            mask[i]  = True

        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return image_t, torch.from_numpy(boxes), torch.from_numpy(mask)


def get_dataloaders(dataset_root, batch_size=8, group_size=12,
                    val_per_group=2, random_seed=42, num_workers=4):
    shared = dict(group_size=group_size, val_per_group=val_per_group,
                  random_seed=random_seed)
    train_ds = CraterCenterDataset(dataset_root, mode="train", **shared)
    val_ds   = CraterCenterDataset(dataset_root, mode="val",   **shared)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


if __name__ == "__main__":
    ROOT = "../../LINNA-Crater/LunarLanding_Dataset/LunarLanding_Dataset"
    ds = CraterCenterDataset(ROOT, mode="train")
    img, boxes, mask = ds[0]
    n = mask.sum().item()
    print(f"image : {img.shape}")
    print(f"craters in this sample: {n}")
    for i in range(n):
        cx, cy, r = boxes[i].tolist()
        print(f"  cx={cx:.3f}  cy={cy:.3f}  r_norm={r:.4f}  r_px={r*IMG_SIZE:.1f}")
