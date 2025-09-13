# CLI prediction
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys

model = tf.keras.models.load_model("models/marathi_ocr_model.h5")

labels = [
    "अ","आ","इ","ई","उ","ऊ","ए","ऐ","ओ","औ","क","ख","ग","घ","च","छ","ज","झ","ञ",
    "ट","ठ","ड","ढ","ण","त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","व",
    "श","ष","स","ह","ळ","क्ष","ज्ञ"
]

def predict_character(img_path):
    img = image.load_img(img_path, target_size=(32,32), color_mode="grayscale")
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = np.argmax(model.predict(arr), axis=1)[0]
    return labels[pred]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        print("Predicted:", predict_character(sys.argv[1]))
