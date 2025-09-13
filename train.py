import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json

# ------------------------
# Paths
# ------------------------
project_root = Path(__file__).parent
train_path = project_root / "data/train"
val_path = project_root / "data/val"
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

IMAGE_SIZE = (64, 64)


folder_to_char = {
    "a_": "अ", "aa": "आ", "i": "इ", "ii": "ई",
    "u": "उ", "uu": "ऊ", "e": "ए", "ai": "ऐ",
    "o": "ओ", "au": "औ", "k": "क", "kh": "ख",
    "g": "ग", "gh": "घ", "ch": "च", "chh": "छ",
    "j": "ज", "jh": "झ", "ny": "ञ", "tt": "ट",
    "tth": "ठ", "dd": "ड", "ddh": "ढ", "nn": "ण",
    "t": "त", "th": "थ", "d": "द", "dh": "ध",
    "n": "न", "p": "प", "ph": "फ", "b": "ब",
    "bh": "भ", "m": "म", "y": "य", "r": "र",
    "l": "ल", "v": "व", "sh": "श", "shh": "ष",
    "s": "स", "h": "ह", "ll": "ळ", "ksh": "क्ष",
    "jnya": "ज्ञ"
}


# ------------------------
# Load images manually
# ------------------------
def load_images_labels(path: Path):
    images = []
    labels = []
    # Make sure class folders are Paths
    class_dirs = sorted([f for f in path.iterdir() if f.is_dir()])
    class_names = [f.name for f in class_dirs]
    
    for idx, folder in enumerate(class_dirs):
        for img_file in folder.glob("*.png"):
            img = load_img(img_file, color_mode="grayscale", target_size=IMAGE_SIZE)
            arr = img_to_array(img)/255.0
            images.append(arr)
            labels.append(idx)
        for img_file in folder.glob("*.jpg"):
            img = load_img(img_file, color_mode="grayscale", target_size=IMAGE_SIZE)
            arr = img_to_array(img)/255.0
            images.append(arr)
            labels.append(idx)
    return np.array(images), np.array(labels), class_names

X_train, y_train, class_names = load_images_labels(train_path)
X_val, y_val, _ = load_images_labels(val_path)

num_classes = len(class_names)
print("Classes:", class_names)
print("Train images:", X_train.shape, "Val images:", X_val.shape)

# ------------------------
# CNN Model
# ------------------------
model = models.Sequential([
    layers.Input(shape=(64,64,1)),
    layers.Conv2D(32,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.Dense(num_classes,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------
# Train
# ------------------------
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=[callback],
    shuffle=True
)

# ------------------------
# Save model + mapping
# ------------------------
model.save(models_dir / "marathi_ocr_model.h5")
with open(models_dir / "index_to_char.json","w",encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print("✅ Training complete. Model + mapping saved!")
