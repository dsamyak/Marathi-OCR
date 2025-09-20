import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import json
import sys

# -------------------------
# Paths
# -------------------------
project_root = Path(__file__).resolve().parent
train_path = project_root / "data" / "train"
val_path   = project_root / "data" / "val"
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

if not train_path.exists() or not val_path.exists():
    print("ERROR: data/train or data/val folder not found. Make sure you run this from project root.")
    sys.exit(1)

# -------------------------
# Parameters
# -------------------------
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# -------------------------
# Load datasets (will infer folder order)
# -------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(train_path),
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(val_path),
    image_size=IMAGE_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int"
)

folder_names = train_ds.class_names
print("Folder order discovered by TF:", folder_names)

# -------------------------
# folder_to_char: romanized -> actual Marathi character
# (Covers all folder names you posted; includes 'ng')
# -------------------------
folder_to_char = {
    "a_":"अ","aa":"आ","i":"इ","ii":"ई","u":"उ","uu":"ऊ",
    "e":"ए","ai":"ऐ","o":"ओ","au":"औ","k":"क","kh":"ख",
    "g":"ग","gh":"घ","ng":"ङ","ch":"च","chh":"छ","j":"ज","jh":"झ",
    "ny":"ञ","tt":"ट","tth":"ठ","dd":"ड","ddh":"ढ","nn":"ण",
    "t":"त","th":"थ","d":"द","dh":"ध","n":"न","p":"प","ph":"फ",
    "b":"ब","bh":"भ","m":"म","y":"य","r":"र","l":"ल","v":"व",
    "sh":"श","shh":"ष","s":"स","h":"ह","ll":"ळ","ksh":"क्ष","jnya":"ज्ञ"
}

# Build index_to_char according to folder order; fallback if missing
index_to_char = []
missing = []
for name in folder_names:
    if name in folder_to_char:
        index_to_char.append(folder_to_char[name])
    else:
        missing.append(name)
        # Fallback: use the folder name itself as placeholder (so training continues)
        index_to_char.append(name)

if missing:
    print("\nWARNING: The following folders were NOT found in folder_to_char mapping:")
    for m in missing:
        print("  -", m)
    print("These folder names will be used as-is in index_to_char.json as placeholders.")
    print("If you want actual Marathi characters for these, add them to folder_to_char in train.py.\n")

num_classes = len(index_to_char)
print("Final index -> character mapping (length {}):".format(num_classes))
print(index_to_char)

# Save mapping (so GUI can load index_to_char.json)
with open(models_dir / "index_to_char.json","w",encoding="utf-8") as f:
    json.dump(index_to_char, f, ensure_ascii=False, indent=2)

# -------------------------
# Data augmentation + perf
# -------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.08, 0.08),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------------------------
# Improved CNN
# -------------------------
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),

    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------
# Callbacks & training
# -------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks
)

# -------------------------
# Save model
# -------------------------
model.save(models_dir / "marathi_ocr_model.h5")
print("✅ Training complete. Saved model and mapping to:", models_dir)
