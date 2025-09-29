# app/gui.py
# -*- coding: utf-8 -*-
"""
Fully functional Marathi OCR GUI
Features:
- Drawing canvas (dynamic brush)
- Top-3 predictions with confidence
- Save drawing as PNG
- Robust preprocessing (auto-invert, resize)
- Compatible with Pillow >=10
"""

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "marathi_ocr_model.h5"
MAPPING_PATH = MODELS_DIR / "index_to_char.json"

# -------------------------
# Constants
# -------------------------
CANVAS_SIZE = 280
PREVIEW_SIZE = 64
DEFAULT_BRUSH = 12
DRAW_COLOR = "black"
BG_COLOR = "white"
TOP_K = 3

# -------------------------
# Global lazy-loaded model
# -------------------------
_model = None
index_to_char = None

def load_model_and_mapping():
    global _model, index_to_char
    if _model is None:
        if not MODEL_PATH.exists() or not MAPPING_PATH.exists():
            raise FileNotFoundError("Model or index_to_char.json not found. Train the model first.")
        _model = tf.keras.models.load_model(str(MODEL_PATH))
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            index_to_char = json.load(f)
    return _model, index_to_char

# -------------------------
# Preprocessing
# -------------------------
def preprocess_image(pil_img):
    """Preprocess image for prediction"""
    img = pil_img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32)/255.0
    if arr.mean() < 0.5:
        arr = 1.0 - arr
    arr = np.expand_dims(arr, axis=(0,-1))
    return arr

# -------------------------
# GUI Class
# -------------------------
class MarathiOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marathi OCR")
        self.root.resizable(False, False)

        # Canvas
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BG_COLOR, cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_pos)

        # PIL image for drawing
        self.image1 = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw_obj = ImageDraw.Draw(self.image1)

        # Brush size slider
        self.brush_size = tk.IntVar(value=DEFAULT_BRUSH)
        ttk.Label(root, text="Brush Size").grid(row=1, column=0)
        self.brush_slider = ttk.Scale(root, from_=1, to=50, variable=self.brush_size, orient=tk.HORIZONTAL)
        self.brush_slider.grid(row=1, column=1)

        # Buttons
        ttk.Button(root, text="Predict", command=self.predict).grid(row=1, column=2, padx=5)
        ttk.Button(root, text="Clear", command=self.clear_canvas).grid(row=1, column=3, padx=5)
        ttk.Button(root, text="Save", command=self.save_image).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(root, text="Quit", command=root.destroy).grid(row=2, column=3, padx=5, pady=5)

        # Prediction labels
        self.pred_label = tk.Label(root, text="Draw a Marathi character", font=("Arial", 16))
        self.pred_label.grid(row=3, column=0, columnspan=4, pady=10)

        self.topk_label = tk.Label(root, text="", font=("Arial", 12))
        self.topk_label.grid(row=4, column=0, columnspan=4)

        self.last_x, self.last_y = None, None

    # -------------------------
    # Drawing methods
    # -------------------------
    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x is None:
            self.last_x, self.last_y = x, y
        size = self.brush_size.get()
        self.canvas.create_line(self.last_x, self.last_y, x, y,
                                width=size, fill=DRAW_COLOR, capstyle=tk.ROUND, smooth=True)
        self.draw_obj.line([self.last_x, self.last_y, x, y], fill=0, width=int(size))
        self.last_x, self.last_y = x, y

    def reset_last_pos(self, event=None):
        self.last_x, self.last_y = None, None

    # -------------------------
    # Functional buttons
    # -------------------------
    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_obj.rectangle([0,0,CANVAS_SIZE,CANVAS_SIZE], fill=255)
        self.pred_label.config(text="Draw a Marathi character")
        self.topk_label.config(text="")

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files","*.png")])
        if file_path:
            self.image1.save(file_path)
            messagebox.showinfo("Saved", f"Image saved to {file_path}")

    def predict(self):
        # Check empty canvas
        if np.array(self.image1).mean() > 250:
            messagebox.showwarning("Empty Canvas", "Please draw something before predicting!")
            return

        try:
            model, mapping = load_model_and_mapping()
        except Exception as e:
            messagebox.showerror("Model Error", str(e))
            return

        arr = preprocess_image(self.image1)
        preds = model.predict(arr)[0]
        top_idx = np.argsort(preds)[::-1][:TOP_K]

        top_chars = []
        for i in top_idx:
            char = mapping.get(str(i), str(i))
            conf = preds[i]*100
            top_chars.append(f"{char}: {conf:.2f}%")

        best_char = mapping.get(str(top_idx[0]), str(top_idx[0]))
        self.pred_label.config(text=f"Predicted: {best_char}")
        self.topk_label.config(text="Top predictions: " + " | ".join(top_chars))

# -------------------------
# Run GUI
# -------------------------
def run():
    root = tk.Tk()
    app = MarathiOCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    run()
