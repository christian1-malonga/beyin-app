"""Rule-based textual justification (why tumour / why no abnormality).

Not a diagnosis — explains what the model saw: confidence, margin,
entropy, and where Grad-CAM focused.
"""
import numpy as np


CLASS_INFO = {
    "glioma": "infiltrative lesion often with irregular borders",
    "meningioma": "typically extra-axial, well-circumscribed, dural-based lesion",
    "pituitary": "sellar / suprasellar region mass",
    "notumor": "no focal abnormal mass signal",
}

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def justify(probs, classes, cam_stats, t_low=0.55, t_high=0.80):
    order = np.argsort(probs)[::-1]
    top, second = order[0], order[1]
    p1, p2 = float(probs[top]), float(probs[second])
    margin = p1 - p2
    entropy = float(-(probs * np.log(probs + 1e-9)).sum())
    label = classes[top]
    focus = cam_stats.get("focus", "central")
    spread = cam_stats.get("spread", 0.0)

    if p1 < t_low or margin < 0.10:
        verdict = "UNCERTAIN — needs human review"
        why = (
            f"Top class '{label}' only {p1:.2f} with small margin {margin:.2f} over "
            f"'{classes[second]}' ({p2:.2f}), entropy {entropy:.2f}. "
            f"Model focus was {focus} (spread {spread:.2f}). Do not trust — re-acquire / refer."
        )
    elif label == "notumor" and p1 >= t_low:
        conf = "high" if p1 >= t_high else "moderate"
        verdict = f"NO ABNORMALITY ({conf} confidence)"
        why = (
            f"No tumour pattern won with {p1:.2f} vs {p2:.2f} for '{classes[second]}' "
            f"(margin {margin:.2f}, entropy {entropy:.2f}). Grad-CAM shows diffuse/low "
            f"{focus} activation (spread {spread:.2f}), consistent with "
            f"{CLASS_INFO['notumor']}. Still requires radiologist confirmation."
        )
    else:
        conf = "high" if p1 >= t_high else "moderate"
        verdict = f"{label.upper()} suspected ({conf} confidence)"
        why = (
            f"'{label}' scored {p1:.2f} vs {p2:.2f} for '{classes[second]}' "
            f"(margin {margin:.2f}, entropy {entropy:.2f}). Model attended to a {focus} "
            f"region (spread {spread:.2f}), consistent with {CLASS_INFO.get(label, 'tumour pattern')}. "
            "This is an assistive cue only — confirm with full series + specialist."
        )
    details = (
        f"probs={ {c: round(float(p),3) for c,p in zip(classes, probs)} } | "
        f"margin={margin:.3f} entropy={entropy:.3f} focus={focus} spread={spread:.3f}"
    )
    return verdict, why, details
