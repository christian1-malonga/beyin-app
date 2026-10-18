"""CPU inference: predict + Grad-CAM + justification. CLI + importable."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `python src/infer.py` and `from src.infer`
import numpy as np
import torch
import yaml

from model import load_checkpoint
from dataset import single_image_tensor
from gradcam import GradCAM, overlay_cam
from justify import justify, softmax


def load_cfg(p="config.yaml"):
    try:
        with open(p) as f: return yaml.safe_load(f)
    except FileNotFoundError:
        return {"model": {"img_size": 224}, "infer": {"threshold_low": .55, "threshold_high": .80}}

def predict(image_path, checkpoint="outputs/best.pt", config="config.yaml", save_heatmap=None):
    cfg = load_cfg(config)
    size = cfg["model"]["img_size"]
    t_low = cfg["infer"]["threshold_low"]; t_high = cfg["infer"]["threshold_high"]
    device = torch.device("cpu")  # force CPU so it runs smoothly anywhere
    model, classes = load_checkpoint(checkpoint, device="cpu")
    x, pil = single_image_tensor(image_path, size)
    g = GradCAM(model)
    cam, cls_idx, logits = g(x)
    probs = softmax(logits.numpy()[0])
    heat, stats = overlay_cam(pil, cam, size=size)
    verdict, why, details = justify(probs, classes, stats, t_low, t_high)
    if save_heatmap:
        from PIL import Image
        Image.fromarray(heat).save(save_heatmap)
    return {
        "predicted": classes[cls_idx],
        "verdict": verdict,
        "why": why,
        "details": details,
        "probs": {c: float(p) for c, p in zip(classes, probs)},
        "heatmap": heat,  # numpy HxWx3
    }

def main():
    ap = argparse.ArgumentParser(description="Beyin App CPU inference")
    ap.add_argument("image", help="path to MRI jpg/png")
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--heatmap-out", default=None)
    a = ap.parse_args()
    r = predict(a.image, a.checkpoint, a.config, a.heatmap_out)
    print(f"Predicted: {r['predicted']}")
    print(f"Verdict: {r['verdict']}")
    print(f"Why: {r['why']}")
    print(r["details"])

if __name__ == "__main__":
    main()
