import os

classes = ["अ", "आ", "इ"]   # add every character you need

for split in ["train", "val"]:
    for c in classes:
        path = os.path.join("data", split, c)
        os.makedirs(path, exist_ok=True)
        print("Created:", path)
