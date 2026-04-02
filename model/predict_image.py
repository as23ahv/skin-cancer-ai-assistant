import os
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

import sys
from pathlib import Path

#Make project root importable
ROOT_DIR = Path(__file__).resolve().parents[1]  # Final Year Project/
sys.path.append(str(ROOT_DIR))

from model.gradcam import build_grad_model, make_gradcam, overlay_heatmap

MODEL_PATH = "model/skin_cancer_model_v2.keras"
LABELS_PATH = "model/labels.json"
OUT_DIR = "outputs/gradcam_single"
IMG_SIZE = (224, 224)

def load_labels():
    with open(LABELS_PATH, "r") as f:
        m = json.load(f)
    return [m[str(i)] for i in range(len(m))]

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    return img, arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Full path to image")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels()
    grad_model = build_grad_model(model)

    original_pil, img_array = preprocess(args.img)
    img_tensor = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(preds))

    top3_idx = np.argsort(preds)[::-1][:3]

    print("\n✅ Prediction Results")
    print(f"Top-1: {labels[pred_idx]} ({preds[pred_idx]*100:.2f}%)")
    print("Top-3:")
    for i in top3_idx:
        print(f"  - {labels[i]}: {preds[i]*100:.2f}%")

    heatmap, _, _ = make_gradcam(grad_model, img_array, class_index=pred_idx)
    overlay = overlay_heatmap(original_pil, heatmap)

    base_name = os.path.splitext(os.path.basename(args.img))[0]
    out_path = os.path.join(OUT_DIR, f"gradcam_{labels[pred_idx]}_{base_name}.png")
    overlay.save(out_path)

    print(f"\n🔥 Saved Grad-CAM to: {out_path}\n")

if __name__ == "__main__":
    main()
