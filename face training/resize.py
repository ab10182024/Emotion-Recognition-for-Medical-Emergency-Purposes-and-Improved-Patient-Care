import os
import cv2
import numpy as np
from pathlib import Path
import shutil

# Define paths
INPUT_DIR = "C:/Users/Alex/Desktop/augmented_face_data"  # Directory containing original 300x300 images
OUTPUT_DIR = "C:/Users/Alex/Desktop/resized_face_data"        # Directory where resized 224x224 images will be saved

# Define original and target sizes
ORIGINAL_SIZE = (300, 300)  # Original image size (Width x Height)
TARGET_SIZE = (224, 224)    # Target size for machine learning (Width x Height)

# Define emotions and severity levels
EMOTIONS = ["Pain", "Shock", "Unconsciousness", "Confusion", "Aggression", "Panic"]
LEVELS = ["Mild", "Moderate", "Severe"]

def resize_image(image_path, output_path):
    """Resize an image to 224x224 and save it to the output path"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error reading image: {image_path}")
            return False
        
        # Resize the image to 224x224
        resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the resized image
        cv2.imwrite(output_path, resized_img)
        return True
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def process_all_images():
    """Process all images in the input directory structure"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create subdirectories for each emotion and severity level
    for emotion in EMOTIONS:
        for level in LEVELS:
            os.makedirs(os.path.join(OUTPUT_DIR, f"{emotion}_{level}"), exist_ok=True)
    
    # Initialize counters
    total_images = 0
    successfully_resized = 0
    
    # Process each emotion and severity level
    for emotion in EMOTIONS:
        for level in LEVELS:
            # Source and destination directories
            src_dir = os.path.join(INPUT_DIR, f"{emotion}_{level}")
            dst_dir = os.path.join(OUTPUT_DIR, f"{emotion}_{level}")
            
            # Skip if source directory doesn't exist
            if not os.path.exists(src_dir):
                print(f"Warning: Source directory not found: {src_dir}")
                continue
            
            # Process all images in the directory
            for file_name in os.listdir(src_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    total_images += 1
                    src_path = os.path.join(src_dir, file_name)
                    dst_path = os.path.join(dst_dir, file_name)
                    
                    if resize_image(src_path, dst_path):
                        successfully_resized += 1
                        print(f"Resized: {src_path} -> {dst_path}")
    
    # Print summary
    print(f"\nResizing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Successfully resized: {successfully_resized}")
    print(f"Failed: {total_images - successfully_resized}")

if __name__ == "__main__":
    print("Starting image resizing process...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Resizing images from {ORIGINAL_SIZE} to {TARGET_SIZE}")
    process_all_images()