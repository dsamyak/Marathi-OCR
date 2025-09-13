import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

# Automatically locate models folder in parent directory
project_root = Path(__file__).resolve().parent.parent
models_dir = project_root / "models"
model_path = models_dir / "marathi_ocr_model.h5"
mapping_path = models_dir / "index_to_char.json"

# Check if files exist
if not model_path.exists() or not mapping_path.exists():
    messagebox.showerror("Error", f"Model or mapping not found!\nExpected at:\n{model_path}\n{mapping_path}")
    raise FileNotFoundError("Model or mapping file not found.")

# Load model and mapping
model = tf.keras.models.load_model(model_path)
with open(mapping_path, "r", encoding="utf-8") as f:
    index_to_char = json.load(f)

# ------------------ GUI ------------------
class MarathiOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Marathi OCR")
        self.root.geometry("450x550")

        tk.Label(root, text="Marathi OCR", font=("Arial", 20, "bold")).pack(pady=10)

        # Drawing canvas
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (280,280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Predict Drawn", command=self.predict_drawn, bg="#4caf50", fg="white", width=15).grid(row=0,column=0,padx=5)
        tk.Button(btn_frame, text="Upload Image", command=self.upload_image, bg="#2196f3", fg="white", width=15).grid(row=0,column=1,padx=5)
        tk.Button(btn_frame, text="Clear", command=self.clear, bg="#f44336", fg="white", width=15).grid(row=0,column=2,padx=5)

        self.label = tk.Label(root, text="Draw or upload a character", font=("Arial", 16))
        self.label.pack(pady=10)

        self.uploaded_img = None

    # Draw on canvas
    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    # Clear canvas
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280,280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.uploaded_img = None
        self.label.config(text="Draw or upload a character")

    # Predict drawn character
    def predict_drawn(self):
        img = self.image.resize((64,64))
        arr = 255 - np.array(img)  # invert colors
        arr = arr / 255.0
        arr = arr.reshape(1,64,64,1)
        pred = model.predict(arr)
        idx = int(np.argmax(pred))
        char = index_to_char[idx]
        confidence = np.max(pred) * 100
        self.label.config(text=f"Prediction: {char} ({confidence:.2f}%)")

    # Predict uploaded image
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files","*.png;*.jpg;*.jpeg")])
        if file_path:
            self.uploaded_img = Image.open(file_path).convert("L").resize((64,64))
            arr = 255 - np.array(self.uploaded_img)
            arr = arr / 255.0
            arr = arr.reshape(1,64,64,1)
            pred = model.predict(arr)
            idx = int(np.argmax(pred))
            char = index_to_char[idx]
            confidence = np.max(pred) * 100
            self.label.config(text=f"Prediction: {char} ({confidence:.2f}%)")

# Run the app
if __name__=="__main__":
    root = tk.Tk()
    MarathiOCRApp(root)
    root.mainloop()
