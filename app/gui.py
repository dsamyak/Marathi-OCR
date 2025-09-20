import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

# -------------------------
# Load model + mapping
# -------------------------
project_root = Path(__file__).parent
models_dir = project_root / "models"

model_path = models_dir / "marathi_ocr_model.h5"
mapping_path = models_dir / "index_to_char.json"

model = tf.keras.models.load_model(model_path)
with open(mapping_path,"r",encoding="utf-8") as f:
    index_to_char = json.load(f)

# -------------------------
# GUI
# -------------------------
class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marathi OCR")
        self.root.geometry("400x500")

        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack(pady=10)

        self.image = Image.new("L",(280,280),255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        tk.Button(root, text="Predict", command=self.predict).pack(pady=5)
        tk.Button(root, text="Clear", command=self.clear).pack(pady=5)

        self.label = tk.Label(root, text="Draw a character", font=("Arial",16))
        self.label.pack(pady=10)

    def paint(self,event):
        x,y = event.x,event.y
        r=8
        self.canvas.create_oval(x-r,y-r,x+r,y+r,fill="black")
        self.draw.ellipse([x-r,y-r,x+r,y+r],fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L",(280,280),255)
        self.draw = ImageDraw.Draw(self.image)
        self.label.config(text="Draw a character")

    def predict(self):
        img = self.image.resize((64,64))
        arr = 255 - np.array(img)  # invert colors
        arr = arr / 255.0
        arr = arr.reshape(1,64,64,1)
        pred = model.predict(arr)
        idx = int(np.argmax(pred))
        char = index_to_char[idx]
        confidence = np.max(pred) * 100
        self.label.config(text=f"Prediction: {char} ({confidence:.2f}%)")

# -------------------------
# Run
# -------------------------
if __name__=="__main__":
    root=tk.Tk()
    OCRApp(root)
    root.mainloop()
