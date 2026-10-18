"""Train MobileNetV2 on brain MRI ImageFolder. Free Colab/Kaggle T4 safe, CPU fallback."""
import argparse, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `python src/train.py` and `from src.train`
import yaml
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def load_cfg(p):
    with open(p) as f: return yaml.safe_load(f)

def _autocast(enabled: bool):
    """Compatible autocast context: torch.amp (2.x) or torch.cuda.amp (1.x)."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _make_scaler(enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def run_epoch(model, loader, crit, opt, device, train=True, scaler=None):
    model.train(train)
    tot, pred_all, true_all = 0.0, [], []
    for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
        x, y = x.to(device), y.to(device)
        if train:
            opt.zero_grad()
            with _autocast(enabled=(scaler is not None)):
                out = model(x); loss = crit(out, y)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        else:
            with torch.no_grad():
                out = model(x); loss = crit(out, y)
        tot += loss.item() * len(y)
        pred_all += out.argmax(1).cpu().tolist()
        true_all += y.cpu().tolist()
    n = len(loader.dataset)
    acc = accuracy_score(true_all, pred_all)
    f1 = f1_score(true_all, pred_all, average="macro", zero_division=0)
    return tot / max(n,1), acc, f1

def main(cfg_path="config.yaml"):
    cfg = load_cfg(cfg_path)
    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    train_ld, val_ld, classes = build_dataloaders(
        cfg["data"]["root"], cfg["model"]["img_size"],
        cfg["train"]["batch_size"], cfg["train"]["val_split"],
        cfg["train"]["num_workers"], cfg["train"]["seed"])
    print(f"Classes: {classes} | train={len(train_ld.dataset)} val={len(val_ld.dataset)}")
    model = build_model(len(classes), cfg["model"]["pretrained"], cfg["model"]["dropout"]).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    use_amp = bool(cfg["train"]["amp"]) and device == "cuda"
    scaler = _make_scaler(use_amp)

    out = Path(cfg["train"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    best, patience, wait = 0.0, cfg["train"]["early_stop_patience"], 0
    for ep in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_ld, crit, opt, device, True, scaler)
        va_loss, va_acc, va_f1 = run_epoch(model, val_ld, crit, None, device, False)
        sched.step()
        print(f"[{ep}/{cfg['train']['epochs']}] loss {tr_loss:.3f}/{va_loss:.3f} "
              f"acc {tr_acc:.3f}/{va_acc:.3f} f1 {tr_f1:.3f}/{va_f1:.3f} ({time.time()-t0:.0f}s)")
        if va_f1 > best:
            best, wait = va_f1, 0
            torch.save({"model": model.state_dict(), "classes": classes, "cfg": cfg}, out / "best.pt")
            print(f"  -> saved best (f1={best:.3f})")
        else:
            wait += 1
            if wait >= patience:
                print("Early stop."); break
    print(f"Done. Best val F1={best:.3f} -> {out/'best.pt'}")
    # export ONNX for fast CPU inference
    try:
        m = build_model(len(classes), pretrained=False)
        ck = torch.load(out / "best.pt", map_location="cpu")
        m.load_state_dict(ck["model"]); m.eval()
        dummy = torch.randn(1,3,cfg["model"]["img_size"],cfg["model"]["img_size"])
        torch.onnx.export(m, dummy, out/"best.onnx", input_names=["input"],
                          output_names=["logits"], opset_version=14)
        print(f"ONNX exported -> {out/'best.onnx'}")
    except Exception as e:
        print(f"ONNX skip: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
