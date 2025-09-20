# predict.py
import sys
from predictor import predict_character

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        char, conf = predict_character(sys.argv[1])
        print(f"Predicted: {char} ({conf:.2f}%)")
