import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
# INPUT_DIR = "C:/Users/Alex/Desktop/original_face_images"  # Directory containing uncropped images
#OUTPUT_DIR = "C:/Users/Alex/Desktop/raw_face_data"        # Directory where cropped faces will be saved


INPUT_DIR = "C:/Users/Alex/Desktop/original_face_images"
OUTPUT_DIR = "C:/Users/Alex/Desktop/raw_face_data" 



# Define target size for all cropped faces
FACE_SIZE = (300, 300)  # Width x Height

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define emotions and severity levels as in the augmentation script
# EMOTIONS = ["Pain", "Shock", "Unconsciousness", "Confusion", "Aggression", "Panic"]
EMOTIONS = ["Unconsciousness"]
LEVELS = ["Mild", "Moderate", "Severe"]

# Create subdirectories for each emotion and severity level
for emotion in EMOTIONS:
    for level in LEVELS:
        os.makedirs(os.path.join(OUTPUT_DIR, f"{emotion}_{level}"), exist_ok=True)

# Load pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Optional: Load more robust face detector if available
try:
    from mtcnn import MTCNN
    detector = MTCNN()
    use_mtcnn = True
    print("Using MTCNN for face detection")
except ImportError:
    use_mtcnn = False
    print("Using Haar Cascade for face detection. For better results, install MTCNN: pip install mtcnn")

def detect_crop_and_resize_face(image_path, output_path, padding=0.2):
    """
    Detect the face in an image, crop it with padding, and resize to 300x300
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the cropped face
        padding: Padding around the face as a fraction of face size
    
    Returns:
        True if face was detected and saved, False otherwise
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return False
    
    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_detected = False
    
    # Try MTCNN first if available (more accurate)
    if use_mtcnn:
        results = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results and len(results) > 0:
            # Use the face with highest confidence
            result = max(results, key=lambda x: x['confidence'])
            x, y, w, h = result['box']
            face_detected = True
    
    # Fall back to Haar Cascade if MTCNN failed or isn't available
    if not face_detected:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            # Use the largest face found
            max_area = 0
            max_face = None
            for (x, y, w, h) in faces:
                if w * h > max_area:
                    max_area = w * h
                    max_face = (x, y, w, h)
            
            if max_face:
                x, y, w, h = max_face
                face_detected = True
    
    if face_detected:
        # Add padding
        padding_x = int(w * padding)
        padding_y = int(h * padding)
        
        # Calculate coordinates with padding
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(img.shape[1], x + w + padding_x)
        y2 = min(img.shape[0], y + h + padding_y)
        
        # Crop the face
        face_img = original_img[y1:y2, x1:x2]
        
        # Resize to 300x300
        face_img_resized = cv2.resize(face_img, FACE_SIZE, interpolation=cv2.INTER_AREA)
        
        # Save the cropped and resized face
        cv2.imwrite(output_path, face_img_resized)
        return True
    else:
        print(f"No face detected in {image_path}")
        return False

def process_dataset():
    """
    Process all images in the input directory and crop faces
    """
    total_images = 0
    processed_images = 0
    failed_images = 0
    
    # Process each emotion and severity level
    for emotion in EMOTIONS:
        for level in LEVELS:
            source_dir = os.path.join(INPUT_DIR, f"{emotion}_{level}")
            target_dir = os.path.join(OUTPUT_DIR, f"{emotion}_{level}")
            
            # Skip if source directory doesn't exist
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory {source_dir} doesn't exist. Skipping.")
                continue
            
            # Get list of images in the source directory
            image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            print(f"Processing {emotion} - {level}: {len(image_files)} images")
            total_images += len(image_files)
            
            # Process each image
            for img_file in tqdm(image_files, desc=f"{emotion}_{level}"):
                img_path = os.path.join(source_dir, img_file)
                output_path = os.path.join(target_dir, img_file)
                
                # Detect, crop and resize face
                success = detect_crop_and_resize_face(img_path, output_path)
                
                if success:
                    processed_images += 1
                else:
                    failed_images += 1
    
    print(f"Processing complete! Total: {total_images}, Succeeded: {processed_images}, Failed: {failed_images}")
    print(f"All processed images have been resized to {FACE_SIZE[0]}x{FACE_SIZE[1]} pixels")

if __name__ == "__main__":
    process_dataset()