# -*- coding: utf-8 -*-
"""
Marathi OCR CNN Training Script (AMD GPU Optimized)
---------------------------------------------------
✅ TensorFlow + DirectML (for AMD Radeon RX 6500M)
✅ Mixed precision + memory growth
✅ ResNet-inspired CNN
✅ Advanced augmentation + cosine LR scheduling
✅ No multiprocessing/workers issues
"""

import os
import json
import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Add, Activation, SpatialDropout2D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    LearningRateScheduler, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# -------------------------------------------------------------------
# 🔧 GPU Configuration (DirectML + AMD)
# -------------------------------------------------------------------
print("🔍 Checking available devices...")
devices = tf.config.list_physical_devices()
for d in devices:
    print("→", d)

# Enable dynamic memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("⚠️ Could not set memory growth:", e)

# Enable mixed precision (for faster training on AMD GPU)
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"✅ Mixed precision policy set to: {mixed_precision.global_policy()}")

# -------------------------------------------------------------------
# 📁 Paths & Config
# -------------------------------------------------------------------
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 50
INITIAL_LR = 0.001
MIN_LR = 1e-6

# Character mapping
folder_to_char = {
 "a": "अ", "aa": "आ", "i": "इ", "ii": "ई", "u": "उ", "uu": "ऊ",
    "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ"
}

# -------------------------------------------------------------------
# 🧠 Model Definition (ResNet-inspired)
# -------------------------------------------------------------------
def residual_block(x, filters, stride=1):
    shortcut = x
    x = Conv2D(filters, 3, strides=stride, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.1)(x)
    x = Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    return Activation('relu')(x)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 7, strides=2, padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    # Residual stacks
    for f, s, n in [(32,1,2), (64,2,3), (128,2,3), (256,2,2)]:
        for i in range(n):
            x = residual_block(x, f, stride=s if i==0 else 1)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    return Model(inputs, outputs, name="MarathiOCR_ResNet_Improved")

# -------------------------------------------------------------------
# 📉 Learning Rate Scheduler
# -------------------------------------------------------------------
def cosine_scheduler(initial_lr, min_lr, warmup_epochs, total_epochs):
    def schedule(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
    return LearningRateScheduler(schedule, verbose=1)

# -------------------------------------------------------------------
# 🧩 Data Augmentation
# -------------------------------------------------------------------
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='constant',
        cval=0
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
    return train_gen, val_gen

# -------------------------------------------------------------------
# 🏋️ Training Function
# -------------------------------------------------------------------
def main():
    print("\n🚀 Starting Marathi OCR training on AMD GPU (DirectML)...\n")
    train_gen, val_gen = create_data_generators()

    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    num_classes = train_gen.num_classes

    model = build_model(input_shape, num_classes)
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )

    model.summary()

    log_dir = LOGS_DIR / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = MODELS_DIR / "marathi_ocr_best.h5"

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', save_best_only=True, verbose=1),
        cosine_scheduler(INITIAL_LR, MIN_LR, 5, EPOCHS),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=MIN_LR, verbose=1),
        TensorBoard(log_dir=str(log_dir))
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    final_path = MODELS_DIR / "marathi_ocr_final.h5"
    model.save(str(final_path))
    print(f"✅ Model saved: {final_path}")

    # Save mappings
    class_indices = train_gen.class_indices
    inv = {v: k for k, v in class_indices.items()}
    index_to_char = {str(i): folder_to_char.get(n, n) for i, n in inv.items()}
    with open(MODELS_DIR / "index_to_char.json", "w", encoding="utf-8") as f:
        json.dump(index_to_char, f, ensure_ascii=False, indent=2)

    print(f"📁 Class mapping saved to {MODELS_DIR}/index_to_char.json")
    print(f"📊 TensorBoard logs: tensorboard --logdir={LOGS_DIR}")
    print(f"🏆 Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")

if __name__ == "__main__":
    main()
