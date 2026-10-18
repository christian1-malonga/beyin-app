"""Download Kaggle brain-tumor MRI set into ImageFolder layout (data/)."""
import shutil
from pathlib import Path

SRC_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"

def main(out="data"):
    out = Path(out)
    try:
        import kagglehub
        src = Path(kagglehub.dataset_download(SRC_SLUG))
        print(f"Downloaded to {src}")
    except Exception as e:
        print(f"kagglehub failed: {e}\n"
              "Manual fallback: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset\n"
              "Download + unzip so that data/<class>/*.jpg exists.")
        return
    # dataset root sometimes nested (e.g. Training/<class> + Testing/<class>)
    # Flatten all class dirs into data/<class>/
    for split in ["Training", "Testing", "training", "testing", "train", "test", ""]:
        base = src / split if split else src
        if not base.exists(): continue
        for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
            for cand in [base / cls, base / cls.capitalize()]:
                if cand.is_dir():
                    dest = out / cls
                    dest.mkdir(parents=True, exist_ok=True)
                    for f in cand.iterdir():
                        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                            shutil.copy(f, dest / f.name)
    # also handle flat layout already correct
    counts = {p.name: len(list(p.glob("*"))) for p in out.iterdir() if p.is_dir()} if out.exists() else {}
    print(f"Ready: {out.resolve()} {counts}")

if __name__ == "__main__":
    main()
