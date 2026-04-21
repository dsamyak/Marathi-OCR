import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import fitz  # PyMuPDF
from PIL import Image
import io

def extract_image_from_file(file_storage):
    """
    Reads an uploaded FileStorage object.
    If PDF, converts the first page to an image.
    If Image, reads it as a numpy array in grayscale.
    """
    file_bytes = file_storage.read()
    filename = file_storage.filename.lower()
    
    if filename.endswith('.pdf'):
        # Open PDF from memory
        doc = fitz.open("pdf", file_bytes)
        if len(doc) == 0:
            raise ValueError("PDF is empty")
        
        # Get first page
        page = doc.load_page(0)
        # Render page to an image (pixmap) with higher resolution
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert pixmap to numpy array
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert RGB to Grayscale
        if pix.n == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
            
        return img_gray
    else:
        # It's an image file
        np_img = np.frombuffer(file_bytes, np.uint8)
        img_color = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img_color is None:
            raise ValueError("Invalid image file")
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        return img_gray

def preprocess_signature(img_gray, target_size=(800, 400)):
    """
    Preprocess the grayscale image:
    1. Threshold to binary (black signature, white background).
    2. Find bounding box of the signature to remove empty whitespace.
    3. Crop and resize to target_size.
    4. Return binary image.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Adaptive thresholding or Otsu's thresholding
    # Since background is typically white or light, Otsu works well
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find coordinates of all non-zero (signature) pixels
    coords = cv2.findNonZero(binary)
    
    if coords is not None:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add some padding
        padding = 10
        y_start = max(0, y - padding)
        y_end = min(binary.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(binary.shape[1], x + w + padding)
        
        cropped = binary[y_start:y_end, x_start:x_end]
    else:
        # If no signature found, use original
        cropped = binary
        
    # Resize to standard dimensions for SSIM matching
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    return resized

def calculate_match_score(file1, file2):
    """
    Takes two FileStorage objects, extracts images, preprocesses them,
    and returns a match percentage using SSIM.
    """
    try:
        img1 = extract_image_from_file(file1)
        img2 = extract_image_from_file(file2)
        
        # Preprocess both to standard size binary images
        processed1 = preprocess_signature(img1)
        processed2 = preprocess_signature(img2)
        
        # Calculate Structural Similarity Index
        # Since images are binary and aligned/resized, SSIM works reasonably well
        score, _ = ssim(processed1, processed2, full=True)
        
        # score is between -1 and 1. We want 0 to 100
        # Typically SSIM for completely different binary structures can go negative, 
        # so we clamp it to 0.
        percentage = max(0, score) * 100
        
        # Minor adjustments: If score is very low, let's keep it low. 
        # If score is somewhat high, it's a good match.
        return round(percentage, 2)
        
    except Exception as e:
        print(f"Error processing images: {e}")
        return 0.0
