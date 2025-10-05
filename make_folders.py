import os

classes = [
"ii",
"u",
"uu",
"e",
"ai",
"o",
"au",
"k",
"kh",
"g",
"gh",
"ng",
"ch",
"chh",
"j","jh","ny","tt","tth","dd","ddh","nn","t","th","d","dh","n","p","ph","b","bh","m","y","r","l","v","sh","shh","s","h","ll","ksh","jnya",
]   # add every character you need

for split in ["train", "val"]:
    for c in classes:
        path = os.path.join("data", split, c)
        os.makedirs(path, exist_ok=True)
        print("Created:", path)
