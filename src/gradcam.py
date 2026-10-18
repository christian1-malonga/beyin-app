"""Minimal Grad-CAM for MobileNetV2 (no extra dependency)."""
import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.grads = None
        self.acts = None
        layer = target_layer or model.features[-1]
        layer.register_forward_hook(self._fwd)
        if hasattr(layer, "register_full_backward_hook"):
            layer.register_full_backward_hook(self._bwd)
        else:  # torch < 1.8 fallback
            layer.register_backward_hook(self._bwd)

    def _fwd(self, mod, inp, out):
        self.acts = out.detach()

    def _bwd(self, mod, gin, gout):
        self.grads = gout[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        score = logits[0, class_idx]
        score.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)  # GAP over H,W
        cam = (w * self.acts).sum(1).squeeze(0)
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx, logits.detach()


def overlay_cam(pil_img, cam, alpha=0.45, size=224):
    img = np.array(pil_img.resize((size, size))).astype(np.uint8)
    heat = cv2.resize(cam, (size, size))
    heat = np.uint8(255 * heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = ((1 - alpha) * img + alpha * heat).astype(np.uint8)
    # focus stats for justification
    h, w = cam.shape
    cy, cx = np.unravel_index(cam.argmax(), cam.shape)
    focus = "central" if (0.3 < cy / h < 0.7 and 0.3 < cx / w < 0.7) else "peripheral"
    spread = float((cam > 0.5).mean())  # fraction strongly activated
    return out, {"focus": focus, "spread": spread, "peak_y": float(cy / h), "peak_x": float(cx / w)}
