# CNN model for Marathi character recognition
import tensorflow as tf
from tensorflow.keras import layers, models

def build_marathi_model(num_classes=46):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
