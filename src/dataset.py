"""Dataset helpers: ImageFolder + stratified train/val split, CPU-safe transforms."""
from pathlib import Path
import random
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def build_transforms(img_size=224, train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_dataloaders(data_root, img_size=224, batch_size=32, val_split=0.2,
                      num_workers=2, seed=42):
    import os
    if os.name == "nt":
        # Windows: multiprocessing DataLoader needs __main__ guard and is
        # flaky on low-RAM home machines -> force single-process loading.
        # Colab/Kaggle (Linux) keeps the configured value.
        num_workers = 0
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Data root not found: {root.resolve()}\n"
            "Run: python scripts/download_dataset.py  (or see README for manual layout)"
        )
    full = datasets.ImageFolder(str(root), transform=build_transforms(img_size, True))
    if len(full) == 0:
        raise RuntimeError(f"No images found under {root.resolve()}")
    labels = [s[1] for s in full.samples]
    idx = list(range(len(full)))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_split, random_state=seed, stratify=labels
    )
    # val set uses non-augmented transform
    val_ds_raw = datasets.ImageFolder(str(root), transform=build_transforms(img_size, False))
    train_ds = Subset(full, train_idx)
    val_ds = Subset(val_ds_raw, val_idx)

    pin = False  # keep CPU-safe; DataLoader sets pin only if CUDA later
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, full.classes


def single_image_tensor(path, img_size=224):
    tf = build_transforms(img_size, False)
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0), img
