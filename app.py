"""Beyin App — Gradio UI, runs on CPU only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from PIL import Image
import tempfile
from infer import predict

CKPT = Path("outputs/best.pt")
CFG = "config.yaml"

DISCLAIMER = ("⚠️ Assistive AI only — not a diagnosis. "
              "Confirm with a radiologist and full MRI series.")

def run(img: Image.Image):
    if img is None:
        return "No image", "Upload an MRI slice.", None, DISCLAIMER
    if not CKPT.exists():
        return ("No checkpoint",
                f"outputs/best.pt not found. Train first (see README) or place a checkpoint there. "
                f"Received image size {img.size}.", None, DISCLAIMER)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as t:
        img.convert("RGB").save(t.name)
        r = predict(t.name, str(CKPT), CFG)
    heat = Image.fromarray(r["heatmap"])
    probs_md = "\n".join(f"- {k}: **{v:.3f}**" for k, v in sorted(r["probs"].items(), key=lambda x: -x[1]))
    out_txt = f"### {r['verdict']}\n\n**Why:** {r['why']}\n\n{probs_md}\n\n`{r['details']}`"
    return r["predicted"], out_txt, heat, DISCLAIMER

demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Brain MRI slice (axial)"),
    outputs=[
        gr.Text(label="Predicted class"),
        gr.Markdown(label="Verdict + justification"),
        gr.Image(label="Grad-CAM focus (red = model looked here)"),
        gr.Markdown(label="Safety"),
    ],
    title="Beyin App",
    description="Upload a brain MRI slice.",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
