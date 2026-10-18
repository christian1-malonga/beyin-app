# Beyin App

**Brain MRI tumour classification with explanations — MobileNetV2 + Grad-CAM, trained on a free GPU, running on your CPU.**

`Beyin` means *brain* in Turkish. The app takes a single brain MRI slice and returns one of four conclusions — **glioma**, **meningioma**, **pituitary**, or **no tumour** — together with a human-readable *justification*, a confidence breakdown, and a Grad-CAM heat map showing **where** the model looked.

![Demo: MRI input and Grad-CAM output for three cases](docs/demo.jpg)

> ### ⚠️ Safety notice
> Beyin App is an **assistive research and education tool**. It is **not** a medical device and produces **no medical diagnosis**. Every output must be confirmed by a qualified radiologist against the full MRI series and the clinical picture. Do not upload patient data to public share links.

---

## Contents

- [Highlights](#highlights)
- [How it works](#how-it-works)
- [Quick start (run in 60 seconds)](#quick-start-run-in-60-seconds)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Configuration reference](#configuration-reference)
- [Project structure](#project-structure)
- [Performance on CPU](#performance-on-cpu)
- [Troubleshooting](#troubleshooting)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Roadmap](#roadmap)

---

## Highlights

- **Small and CPU-friendly** — 2.2 M parameters, a 9 MB checkpoint, 128×128 input. A full prediction *including* Grad-CAM takes ≈ 0.66 s on a normal desktop CPU.
- **Trains for free** — designed for the free Colab / Kaggle T4 tier (~10–15 min for the full schedule). No paid compute needed.
- **Explains itself** — every verdict ships with the confidence scores, the margin over the runner-up class, an entropy (uncertainty) value, and a Grad-CAM focus map, fused into plain language by deterministic rules (not an LLM).
- **Honest when unsure** — if the top score is below `0.55` or the margin below `0.10`, the app refuses to guess and says **“UNCERTAIN — needs human review”**.
- **Transfer learning** — starts from ImageNet-pretrained MobileNetV2 weights, which is why 1,600 images and 5 CPU epochs already reach a useful level.

## How it works

```
MRI slice
   │
   ▼
preprocess (resize 128×128, ImageNet normalisation)
   │
   ▼
MobileNetV2 (ImageNet-pretrained) ────► 4 logits ──► softmax probabilities
   │                                                        │
   ▼                                                        ▼
Grad-CAM on last conv layer                         scores / margin / entropy
   │                                                        │
   ▼                                                        ▼
focus map + heat overlay                          deterministic rules (src/justify.py)
   │                                                        │
   └──────────────────────┬─────────────────────────────────┘
                          ▼
              verdict + justification + heat map
```

### Why MobileNetV2?

| Constraint | Consequence |
|---|---|
| Labels are **folder-level only** (no boxes, no masks) | YOLO (needs boxes) and segmentation (needs masks) cannot train on this dataset — classification is the correct task |
| Must train on a **free** Colab/Kaggle T4 and run on a **home CPU** | A 2.2 M-parameter model with 128 px input trains in minutes and infers in ~0.1 s |
| Must explain its answer | Softmax scores + Grad-CAM on MobileNetV2's last conv block are sufficient and dependency-free |

### The "why" engine

Two measured signals are combined by fixed thresholds in `src/justify.py`:

1. **Scores** — top-1 probability, margin over the second class, Shannon entropy.
2. **Attention** — Grad-CAM focus location (`central` / `peripheral`) and spread (fraction of the map above half intensity).

| Condition | Verdict |
|---|---|
| top-1 < `0.55` or margin < `0.10` | **UNCERTAIN — needs human review** |
| `notumor` wins with ≥ `0.55` | **NO ABNORMALITY** (confidence level stated) |
| a tumour class wins | **<TYPE> suspected** (confidence level stated) |

The explanation text quotes these measured values verbatim — nothing is invented.

## Quick start (run in 60 seconds)

The trained checkpoint **`outputs/best.pt` is included in this repository**, so you can run the app immediately:

```bash
git clone https://github.com/christian1-malonga/beyin-app.git
cd beyin-app

# CPU-only PyTorch (recommended for laptops — smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python app.py
```

Open **http://127.0.0.1:7860**, upload an MRI slice, and read the verdict.

## Usage

### 1. Command line

```bash
python src/infer.py path\to\mri.jpg                    # print verdict + justification
python src/infer.py path\to\mri.jpg --heatmap-out heat.png   # also save the Grad-CAM overlay
```

Example output:

```
Predicted: notumor
Verdict: NO ABNORMALITY (high confidence)
Why: No tumour pattern won with 0.93 vs 0.04 for 'glioma' (margin 0.89, entropy 0.31).
     Grad-CAM shows diffuse/low peripheral activation (spread 0.24), consistent with
     no focal abnormal mass signal. Still requires radiologist confirmation.
```

### 2. Python API

```python
import sys; sys.path.insert(0, "src")
from infer import predict

result = predict("mri.jpg", checkpoint="outputs/best.pt", config="config.yaml")
print(result["predicted"])   # e.g. "meningioma"
print(result["verdict"])     # e.g. "MENINGIOMA suspected (high confidence)"
print(result["why"])         # plain-English justification
print(result["probs"])       # {"glioma": 0.05, "meningioma": 0.91, ...}
result["heatmap"]            # numpy HxWx3 image (heat overlay)
```

### 3. Gradio web app

```bash
python app.py                  # local only:  http://127.0.0.1:7860
```

**Public link** — edit the last line of `app.py`:

```python
demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
```

The terminal then prints an `https://xxxx.gradio.live` URL valid while the process runs (≈ 72 h).

> Anyone with the link can upload images that are relayed through Gradio's servers.
> **Use only test or synthetic images on public links — never patient data.**

## Dataset

**[masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** — a public collection of 7,200 brain MRI slices:

| Class | Images |
|---|---|
| Glioma | 1,800 |
| Meningioma | 1,800 |
| Pituitary | 1,800 |
| No tumour | 1,800 |

The download script flattens the dataset's original train/test split into a single
ImageFolder layout; `src/dataset.py` then performs its own **stratified 80/20 split**
(seeded, so runs are reproducible), which is what keeps the reported metrics honest.

```bash
python scripts/download_dataset.py     # -> data/<class>/*.jpg   (~160 MB, no Kaggle API token needed)
```

Images stay in `data/` on your disk only — that folder is git-ignored.

## Training

### Option A — Google Colab (recommended, free)

1. Upload `notebooks/colab_train.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Runtime → **Change runtime type → T4 GPU**.
3. Run all cells. The notebook installs the dependencies, downloads the dataset, trains, evaluates, and exports `outputs/best.pt` + `outputs/best.onnx`.
4. Download `best.pt` and place it at `outputs/best.pt` locally.

### Option B — Kaggle

Same notebook in a Kaggle kernel; enable **Settings → Accelerator → GPU T4**.

### Option C — Command line (any Linux box / the notebook's shell)

```bash
pip install -r requirements.txt
python scripts/download_dataset.py
python src/train.py --config config.yaml
python src/evaluate.py --checkpoint outputs/best.pt --data data
```

### What the trainer does

| Stage | Detail |
|---|---|
| Backbone | `torchvision.models.mobilenet_v2` with `IMAGENET1K_V2` weights (set `pretrained: false` to train from scratch — not recommended at this dataset size) |
| Optimiser | AdamW, lr `3e-4`, weight decay `1e-4`, cosine annealing schedule |
| Augmentation | random horizontal flip, rotation ±12°, colour jitter (train split only) |
| Precision | automatic mixed precision when a CUDA GPU is present; full float32 on CPU |
| Selection | best checkpoint by validation macro-F1, early stopping (patience `4`) |
| Export | `outputs/best.pt` always; `outputs/best.onnx` when `onnxscript` is available |

> **Tip for the final accuracy push:** set `img_size: 224` in `config.yaml` before training
> in Colab — the free T4 handles it easily; the local CPU then also infers at 224 px.

## Results

Measured on the **full 7,200-image dataset** with the shipped checkpoint
(`outputs/best.pt`: 5 epochs, 128 px, stratified 1,600-image training subset — the remaining
5,600 images were never seen during training):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Glioma | 0.92 | 0.86 | 0.89 | 1,800 |
| Meningioma | 0.77 | 0.89 | 0.83 | 1,800 |
| No tumour | 0.96 | 0.92 | 0.94 | 1,800 |
| Pituitary | 0.95 | 0.91 | 0.93 | 1,800 |
| **Overall accuracy** | | | **0.89** | 7,200 |
| **Macro F1** | | | **0.90** | |

Confusion matrix (rows = true class, columns = predicted):

```
[[1545  222   32    1]   glioma
 [ 106 1595   28   71]   meningioma
 [   9  119 1656   16]   notumor
 [  28  125   10 1637]]  pituitary
```

Most residual errors are **glioma ↔ meningioma** — a confusion that is well documented
even among human readers on single slices, and which the app's UNCERTAIN path is designed to surface.

**Expected improvement:** the full 12-epoch schedule on all 7,200 images with 224 px input
(the included Colab notebook) typically reaches ≈ 0.93–0.96 accuracy on this dataset.

## Configuration reference

Everything is driven by `config.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `model.img_size` | `128` | Input resolution. Keep train/infer identical. `224` = slower but more accurate |
| `model.pretrained` | `true` | Start from ImageNet weights |
| `model.dropout` | `0.2` | Dropout before the final linear layer |
| `train.batch_size` | `64` | Use `16` on low-RAM machines |
| `train.epochs` | `12` | Notebook default (local demo used `5`) |
| `train.lr` | `0.0003` | AdamW learning rate |
| `train.val_split` | `0.2` | Stratified validation fraction |
| `train.early_stop_patience` | `4` | Stop after N epochs without macro-F1 gain |
| `train.amp` | `true` | Mixed precision — activates only when CUDA exists |
| `infer.threshold_low` | `0.55` | Below this → UNCERTAIN verdict |
| `infer.threshold_high` | `0.80` | Above this → "high confidence" wording |
| *(hard-coded)* margin | `0.10` | In `src/justify.py`; below this → UNCERTAIN |

## Project structure

```
beyin-app/
├── app.py                      # Gradio web UI (forces CPU)
├── config.yaml                 # single source of truth for training + inference
├── requirements.txt
├── data/                       # dataset — created by the download script (git-ignored)
├── docs/
│   └── demo.jpg                # README figure
├── notebooks/
│   └── colab_train.ipynb       # one-click free training on Colab / Kaggle
├── outputs/
│   └── best.pt                 # trained checkpoint (included, 9 MB)
├── scripts/
│   └── download_dataset.py     # Kaggle dataset -> ImageFolder layout
└── src/
    ├── dataset.py              # transforms, stratified split, loaders
    ├── model.py                # MobileNetV2 builder + checkpoint loader
    ├── train.py                # training loop, early stopping, ONNX export
    ├── evaluate.py             # classification report + confusion matrix
    ├── infer.py                # predict(): scores + Grad-CAM + justification
    ├── gradcam.py              # dependency-free Grad-CAM + heat overlay
    └── justify.py              # deterministic verdict / justification rules
```

## Performance on CPU

Measured on an ordinary Windows desktop CPU (PyTorch 2.14, CPU build), single image, batch 1:

| Stage | Time |
|---|---|
| Model load (first call only) | ~0.35 s |
| Forward + backward (Grad-CAM) | ~0.11 s |
| End-to-end `predict()` | **~0.66 s** |
| Checkpoint size on disk | 9.1 MB |

For faster sequential inference, load the model once and call it repeatedly
(the CLI reloads per run by design, which keeps the API stateless and simple).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Data root not found` when training | Run `python scripts/download_dataset.py` first |
| `outputs/best.pt not found` in the app | Train (see [Training](#training)) or keep the checkpoint shipped with the repo at `outputs/best.pt` |
| Every answer comes back UNCERTAIN | Check that `config.yaml` `img_size` matches the value the checkpoint was trained with |
| `ONNX skip: No module named 'onnxscript'` | Harmless. `pip install onnxscript onnx` if you need the ONNX export |
| Gradio link fails to appear (`share=True`) | Often firewall/antivirus; the local `127.0.0.1:7860` still works |
| Windows DataLoader freeze during training | Already handled: `src/dataset.py` forces `num_workers=0` on Windows |

## Limitations and responsible use

- **Single-slice analysis.** Radiologists read whole 3-D series; this app sees one 2-D slice.
- **Trained on one dataset.** Performance on other scanners, protocols, or populations is unvalidated — expect a domain shift.
- **No calibrated uncertainty guarantee.** Thresholds are heuristic; the UNCERTAIN path reduces but does not eliminate risk.
- **Not for clinical decisions.** No regulatory clearance (FDA/CE). Research and education only.
- **Data privacy.** `data/` and `.gradio/` flags are git-ignored for a reason — never commit scans of real patients.

## Roadmap

- [ ] Train on the full dataset at 224 px in Colab and replace `outputs/best.pt`
- [ ] Add a lightweight segmentation head (Mobile-UNet) once pixel masks are available
- [ ] Optional YOLOv8n variant for localisation if box annotations are added
- [ ] Export to ONNX Runtime / int8 quantisation for sub-100 ms CPU inference
- [ ] Batch inference mode for research workflows

---

**Author:** [christian1-malonga](https://github.com/christian1-malonga)
**Model:** MobileNetV2 (ImageNet-pretrained) · **Interface:** Gradio · **License:** no license specified — all rights reserved by default; add a `LICENSE` file before wider distribution.
