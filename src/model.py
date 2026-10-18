"""MobileNetV2 classifier — ~14MB, ~30ms/image on CPU."""
import torch
from torch import nn
from torchvision import models


def build_model(num_classes=4, pretrained=True, dropout=0.2):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_f, num_classes),
    )
    return m


def load_checkpoint(path, num_classes=4, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    classes = ckpt.get("classes", [f"class_{i}" for i in range(num_classes)])
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, classes
