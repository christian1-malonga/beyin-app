"""Evaluate checkpoint on a val/test ImageFolder."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix
from dataset import build_transforms
from model import load_checkpoint

def main(ckpt, data_root, img_size=224):
    model, classes = load_checkpoint(ckpt, device="cpu")
    ds = datasets.ImageFolder(data_root, transform=build_transforms(img_size, False))
    ld = DataLoader(ds, batch_size=32)
    P, T = [], []
    with torch.no_grad():
        for x, y in ld:
            P += model(x).argmax(1).tolist(); T += y.tolist()
    print(classification_report(T, P, target_names=classes, zero_division=0))
    print("confusion_matrix:\n", confusion_matrix(T, P))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args()
    main(a.checkpoint, a.data, a.img_size)
