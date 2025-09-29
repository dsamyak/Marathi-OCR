# predict.py
# -*- coding: utf-8 -*-
"""
CLI prediction script for Marathi OCR
Features:
- Preprocess input image (64x64, grayscale, auto-invert)
- Show top-3 predictions with confidence
- Handles missing model / mapping
- Compatible with Pillow >=10
"""

import argparse
from pathlib import Path
import json
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "marathi_ocr_model.h5"
MAPPING_PATH = MODELS_DIR / "index_to_char.json"

TOP_K = 3
PREVIEW_SIZE = 64

# -------------------------
# Preprocessing
# -------------------------
def preprocess_image(pil_img):
    img = pil_img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32)/255.0
    if arr.mean() < 0.5:
        arr = 1.0 - arr
    arr = np.expand_dims(arr, axis=(0,-1))
    return arr

# -------------------------
# Load model and mapping
# -------------------------
def load_model_and_mapping():
    if not MODEL_PATH.exists() or not MAPPING_PATH.exists():
        raise FileNotFoundError("Model or index_to_char.json not found. Train the model first.")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        index_to_char = json.load(f)
    return model, index_to_char

# -------------------------
# Prediction
# -------------------------
def predict_image(image_path):
    model, mapping = load_model_and_mapping()
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        return

    img = Image.open(image_path)
    arr = preprocess_image(img)
    preds = model.predict(arr)[0]
    top_idx = np.argsort(preds)[::-1][:TOP_K]

    print(f"\nPredictions for {image_path}:\n")
    for i in top_idx:
        char = mapping.get(str(i), str(i))
        conf = preds[i]*100
        print(f"{char}: {conf:.2f}%")
    print(f"\nMost likely character: {mapping.get(str(top_idx[0]), str(top_idx[0]))}\n")

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict Marathi character from an image.")
    parser.add_argument("image", help="Path to input image (png/jpg)")
    args = parser.parse_args()

    predict_image(args.image)

if __name__ == "__main__":
    main()
