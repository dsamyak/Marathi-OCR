# -*- coding: utf-8 -*-
"""
Train Marathi OCR CNN model.
"""

import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------
# Config
# -------------------------
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

folder_to_char = {
    "ka": "क", "kha": "ख", "ga": "ग", "gha": "घ", "nga": "ङ",
    "cha": "च", "chha": "छ", "ja": "ज", "jha": "झ", "nya": "ञ",
    "ta": "ट", "tha": "ठ", "da": "ड", "dha": "ढ", "na": "ण",
    "pa": "प", "pha": "फ", "ba": "ब", "bha": "भ", "ma": "म",
    "ya": "य", "ra": "र", "la": "ल", "va": "व",
    "sha": "श", "sra": "श्र", "sa": "स", "ha": "ह",
    "a": "अ", "aa": "आ", "i": "इ", "ii": "ई", "u": "उ", "uu": "ऊ",
    "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ"
}

# -------------------------
# Model
# -------------------------
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding="same"),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding="same"),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding="same"),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -------------------------
# Train
# -------------------------
def main():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        str(VAL_DIR),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    num_classes = train_gen.num_classes
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    model = build_model(input_shape, num_classes)
    model.summary()

    checkpoint_path = MODELS_DIR / "marathi_ocr_model.h5"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

    # Save index-to-char mapping
    class_indices = train_gen.class_indices
    inv = {v: k for k,v in class_indices.items()}
    index_to_char = {str(idx): folder_to_char.get(name, name) for idx,name in inv.items()}

    with open(MODELS_DIR/"index_to_char.json", "w", encoding="utf-8") as f:
        json.dump(index_to_char, f, ensure_ascii=False, indent=2)

    print("✅ Training complete. Model & mapping saved.")

if __name__ == "__main__":
    main()
