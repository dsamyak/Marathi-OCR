# predict.py
# -*- coding: utf-8 -*-
"""
CLI prediction for Marathi OCR.
Usage:
    python predict.py /path/to/image.png
Outputs top-K predictions and the most likely character.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "marathi_ocr_model.h5"
MAPPING_FILE = MODELS_DIR / "index_to_char.json"

IMG_SIZE = 64
TOP_K = 3


def preprocess_pil_image_for_model(pil: Image.Image, out_size: int = IMG_SIZE) -> np.ndarray:
    img = pil.convert("L")
    img = ImageOps.invert(img)
    arr = np.array(img)
    thresh = 128
    bin_arr = (arr > thresh).astype("uint8") * 255
    bin_arr = 255 - bin_arr
    coords = np.argwhere(bin_arr > 0)
    if coords.size:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        crop = bin_arr[y0:y1+1, x0:x1+1]
    else:
        crop = np.full((out_size, out_size), 0, dtype=np.uint8)
    crop_pil = Image.fromarray(crop).convert("L")
    crop_pil.thumbnail((out_size, out_size), Image.Resampling.LANCZOS)
    new_img = Image.new("L", (out_size, out_size), 0)
    paste_x = (out_size - crop_pil.width) // 2
    paste_y = (out_size - crop_pil.height) // 2
    new_img.paste(crop_pil, (paste_x, paste_y))
    final = np.array(new_img).astype("float32") / 255.0
    final = np.expand_dims(final, axis=-1)
    return final


def load_model_and_mapping():
    if not MODEL_FILE.exists() or not MAPPING_FILE.exists():
        raise FileNotFoundError("Model or mapping not found; run train.py first.")
    model = tf.keras.models.load_model(str(MODEL_FILE))
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return model, mapping


def predict_image(image_path: Path):
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)
    model, mapping = load_model_and_mapping()
    pil = Image.open(str(image_path))
    arr = preprocess_pil_image_for_model(pil, IMG_SIZE)
    x = np.expand_dims(arr.squeeze(), axis=0) if arr.ndim == 3 else np.expand_dims(arr, axis=0)
    preds = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:TOP_K]
    print(f"\nPredictions for {image_path.name}:\n")
    for i in top_idx:
        char = mapping.get(str(i), str(i))
        print(f"{char}: {preds[i]*100:.2f}%")
    print(f"\nMost likely: {mapping.get(str(top_idx[0]), str(top_idx[0]))}\n")


def main():
    parser = argparse.ArgumentParser(description="Predict Marathi character from image")
    parser.add_argument("image", help="Path to image file (png/jpg)")
    args = parser.parse_args()
    predict_image(Path(args.image))


if __name__ == "__main__":
    main()
