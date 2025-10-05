import os
import io
import json
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageTk
import tkinter as tk
from tkinter import messagebox
import tensorflow as tf

# === Locate model and mapping ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/marathi_ocr_model.h5")
MAPPING_PATH = os.path.join(BASE_DIR, "../models/index_to_char.json")

# === Load model ===
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load model:\n{e}")
    raise SystemExit

# === Load label mapping ===
try:
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        index_to_char = json.load(f)
    print("✅ Mapping file loaded!")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load mapping:\n{e}")
    raise SystemExit


# === Main GUI App ===
class MarathiOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🪶 Marathi OCR Recognition")
        self.root.geometry("400x520")
        self.root.configure(bg="#f9f9f9")

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white", cursor="cross")
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Buttons
        button_frame = tk.Frame(root, bg="#f9f9f9")
        button_frame.pack(pady=5)

        tk.Button(button_frame, text="Predict", command=self.predict,
                  bg="#4CAF50", fg="white", width=12, font=("Arial", 11, "bold")).grid(row=0, column=0, padx=6)
        tk.Button(button_frame, text="Clear", command=self.clear,
                  bg="#F44336", fg="white", width=12, font=("Arial", 11, "bold")).grid(row=0, column=1, padx=6)

        # Result Labels
        self.pred_label = tk.Label(root, text="Draw a Marathi character ✍️",
                                   font=("Arial", 14, "bold"), bg="#f9f9f9", fg="#333")
        self.pred_label.pack(pady=15)

        self.conf_label = tk.Label(root, text="", font=("Arial", 12), bg="#f9f9f9", fg="#555")
        self.conf_label.pack()

        # Mouse event
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Draw a Marathi character ✍️", fg="#333")
        self.conf_label.config(text="")

    def preprocess(self, img):
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # (1, 64, 64, 1)
        return img

    def predict(self):
        try:
            img = self.image.copy()
            x = self.preprocess(img)
            preds = model.predict(x, verbose=0)
            pred_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0])) * 100
            char = index_to_char.get(str(pred_idx), "❓")

            self.pred_label.config(text=f"Predicted: {char}", fg="#2E7D32")
            self.conf_label.config(text=f"Confidence: {confidence:.2f}%", fg="#444")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Something went wrong:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MarathiOCRApp(root)
    root.mainloop()
