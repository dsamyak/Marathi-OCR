# predictor.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path
import json

# Paths
models_dir = Path(__file__).parent / "models"
model_path = models_dir / "marathi_ocr_model.h5"
labels_path = models_dir / "index_to_char.json"

# Load model once
model = tf.keras.models.load_model(model_path)

# Load labels
if labels_path.exists():
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
else:
    labels = ["अ","आ","इ"]  # fallback if mapping not found

def predict_character(img_input):
    """
    img_input: can be a file path (str/Path) or a PIL Image
    returns: predicted character + confidence
    """
    if isinstance(img_input, (str, Path)):
        img = image.load_img(img_input, target_size=(64,64), color_mode="grayscale")
    else:
        # Assume PIL image (used in GUI canvas)
        img = img_input.resize((64,64)).convert("L")

    arr = image.img_to_array(img)
    arr = 255 - arr  # invert colors (black strokes on white background)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)  # shape: (64,64,1)
    arr = np.expand_dims(arr, axis=0)   # shape: (1,64,64,1)

    pred = model.predict(arr, verbose=0)
    idx = int(np.argmax(pred))
    char = labels[idx]
    confidence = float(np.max(pred)) * 100
    return char, confidence
