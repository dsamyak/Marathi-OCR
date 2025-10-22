# predict_improved.py
# -*- coding: utf-8 -*-
"""
IMPROVED CLI prediction for Marathi OCR with better preprocessing
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import tensorflow as tf
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "marathi_ocr_improved.h5"
MAPPING_FILE = MODELS_DIR / "index_to_char.json"

IMG_SIZE = 64
TOP_K = 5

def advanced_preprocess(pil_img: Image.Image, out_size: int = IMG_SIZE) -> np.ndarray:
    """
    Advanced preprocessing pipeline for better accuracy
    """
    # Convert to grayscale
    img = pil_img.convert("L")
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Convert to numpy for OpenCV operations
    img_array = np.array(img)
    
    # Apply bilateral filter to reduce noise while preserving edges
    img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
    
    # Adaptive thresholding for better binarization
    binary = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours and crop to bounding box
    coords = np.argwhere(binary > 0)
    
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        # Add small padding
        pad = 3
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(binary.shape[0], y1 + pad)
        x1 = min(binary.shape[1], x1 + pad)
        
        crop = binary[y0:y1+1, x0:x1+1]
    else:
        crop = np.zeros((out_size, out_size), dtype=np.uint8)
    
    # Convert back to PIL for resizing
    crop_pil = Image.fromarray(crop).convert("L")
    
    # Aspect-preserving resize
    crop_pil.thumbnail((out_size, out_size), Image.Resampling.LANCZOS)
    
    # Create centered image
    new_img = Image.new("L", (out_size, out_size), 0)
    paste_x = (out_size - crop_pil.width) // 2
    paste_y = (out_size - crop_pil.height) // 2
    new_img.paste(crop_pil, (paste_x, paste_y))
    
    # Normalize
    final = np.array(new_img).astype("float32") / 255.0
    
    # Add channel dimension
    final = np.expand_dims(final, axis=-1)
    
    return final

def load_model_and_mapping():
    """Load model and character mapping"""
    if not MODEL_FILE.exists():
        print(f"❌ Model not found: {MODEL_FILE}")
        sys.exit(1)
    
    if not MAPPING_FILE.exists():
        print(f"❌ Mapping file not found: {MAPPING_FILE}")
        sys.exit(1)
    
    model = tf.keras.models.load_model(str(MODEL_FILE))
    
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        idx_to_char = json.load(f)
    
    return model, idx_to_char

def predict_character(model, idx_to_char, img_path: str | Path, top_k: int = TOP_K):
    """
    Predict character from image with confidence scores
    """
    img_path = Path(img_path)
    
    if not img_path.exists():
        print(f"❌ Image not found: {img_path}")
        return None
    
    # Load and preprocess image
    pil_img = Image.open(img_path)
    processed = advanced_preprocess(pil_img)
    
    # Add batch dimension
    batch = np.expand_dims(processed, axis=0)
    
    # Predict
    predictions = model.predict(batch, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        char = idx_to_char.get(str(idx), "?")
        confidence = float(predictions[idx]) * 100
        results.append((char, confidence))
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Predict Marathi character from image (improved version)"
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                       help=f"Show top-K predictions (default: {TOP_K})")
    
    args = parser.parse_args()
    
    print("🔮 Loading improved model...")
    model, idx_to_char = load_model_and_mapping()
    
    print(f"📸 Processing: {args.image_path}")
    results = predict_character(model, idx_to_char, args.image_path, args.top_k)
    
    if results:
        print("\n✨ Predictions:")
        print("-" * 40)
        for i, (char, conf) in enumerate(results, 1):
            symbol = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{symbol} {i}. '{char}' - {conf:.2f}% confidence")
        print("-" * 40)
        print(f"\n🎯 Most likely: '{results[0][0]}' ({results[0][1]:.2f}%)")

if __name__ == "__main__":
    main()
