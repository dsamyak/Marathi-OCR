# train.py
# -*- coding: utf-8 -*-
"""
Training script for Marathi OCR.
- Loads data from data/train and data/val (romanized folder names).
- Applies augmentation, builds a CNN, trains with early stopping.
- Saves models/marathi_ocr_model.h5 and models/index_to_char.json
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Ensure reproducible-ish behavior
tf.random.set_seed(42)
np.random.seed(42)

# -------------------------
# Configuration
# -------------------------
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)     # As required
BATCH_SIZE = 32
EPOCHS = 20

# Example romanized -> Marathi mapping.
# Add more mappings to this dict as you label folders.
# If a folder is missing here, the fallback uses the folder name itself.
folder_to_char = {
    # basic consonants (examples)
    "ka": "क",
    "kha": "ख",
    "ga": "ग",
    "gha": "घ",
    "nga": "ङ",
    "cha": "च",
    "chha": "छ",
    "ja": "ज",
    "jha": "झ",
    "nya": "ञ",
    "ta": "ट",
    "tha": "ठ",
    "da": "ड",
    "dha": "ढ",
    "na": "ण",
    "pa": "प",
    "pha": "फ",
    "ba": "ब",
    "bha": "भ",
    "ma": "म",
    "ya": "य",
    "ra": "र",
    "la": "ल",
    "va": "व",
    "sha": "श",
    "sra": "श्र",  # example conjunct
    "sa": "स",
    "ha": "ह",
    # vowels
    "a": "अ",
    "aa": "आ",
    "i": "इ",
    "ii": "ई",
    "u": "उ",
    "uu": "ऊ",
    "e": "ए",
    "ai": "ऐ",
    "o": "ओ",
    "au": "औ",
}

# -------------------------
# Helpers
# -------------------------
def build_model(input_shape, num_classes):
    """Simple but effective CNN for grayscale character classification."""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def create_index_to_char_map(class_indices):
    """
    Given Keras' class_indices (folder_name -> class_index),
    return index_to_char mapping where index is stringified (so json safe)
    and value is Marathi char (fallback to folder name).
    """
    # invert class_indices to index -> folder name
    inv = {v: k for k, v in class_indices.items()}
    index_to_char = {}
    for idx_int, folder in inv.items():
        # Use mapping if exists; otherwise fallback to folder string
        char = folder_to_char.get(folder, folder)
        index_to_char[str(idx_int)] = char
    return index_to_char


# -------------------------
# Main training flow
# -------------------------
def main(args):
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.10,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Flow from directories: assumed structure data/train/<class_name>/*.png
    print("Loading training data from:", TRAIN_DIR)
    train_generator = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    print("Loading validation data from:", VAL_DIR)
    val_generator = val_datagen.flow_from_directory(
        str(VAL_DIR),
        target_size=IMAGE_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = train_generator.num_classes
    print(f"Number of classes detected: {num_classes}")

    # Build model
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    model = build_model(input_shape, num_classes)
    model.summary()

    # Callbacks
    checkpoint_path = MODELS_DIR / "marathi_ocr_model.h5"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # Fit
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # After training, ensure best weights are saved (ModelCheckpoint did it)
    # Create index_to_char mapping and save it
    class_indices = train_generator.class_indices  # folder -> index
    index_to_char = create_index_to_char_map(class_indices)

    mapping_path = MODELS_DIR / "index_to_char.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False keeps Marathi characters readable in JSON
        json.dump(index_to_char, f, ensure_ascii=False, indent=2)

    print(f"Saved model to: {checkpoint_path}")
    print(f"Saved index_to_char mapping to: {mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Marathi OCR model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    args = parser.parse_args()
    EPOCHS = args.epochs
    main(args)
