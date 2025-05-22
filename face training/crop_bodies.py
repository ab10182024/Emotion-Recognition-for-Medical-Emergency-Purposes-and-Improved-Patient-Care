import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
INPUT_DIR = "C:/Users/Alex/Desktop/original_body_images"  # Directory containing original images
OUTPUT_DIR = "C:/Users/Alex/Desktop/raw_body_data"         # Directory where cropped bodies will be saved

# Define target size for cropped bodies
BODY_SIZE = (300, 300)  # Width x Height

# Emotions and severity levels (same as faces)
EMOTIONS = ["Pain", "Shock", "Unconsciousness", "Confusion", "Aggression", "Panic"]
LEVELS = ["Mild", "Moderate", "Severe"]

# Create output folders if they don't exist
for emotion in EMOTIONS:
    for level in LEVELS:
        os.makedirs(os.path.join(OUTPUT_DIR, f"{emotion}_{level}"), exist_ok=True)

# Load Haar cascade for full body detection
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

if body_cascade.empty():
    print("Warning: Full body Haar cascade not found. Check OpenCV installation.")


def detect_crop_and_resize_body(image_path, output_path, padding=0.15):
    """
    Detect body, crop with padding, resize to 300x300, and save.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return False

    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    if len(bodies) == 0:
        print(f"No body detected in {image_path}")
        return False

    # Pick the largest detected body
    x, y, w, h = max(bodies, key=lambda rect: rect[2] * rect[3])

    # Apply padding
    padding_x = int(w * padding)
    padding_y = int(h * padding)

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(img.shape[1], x + w + padding_x)
    y2 = min(img.shape[0], y + h + padding_y)

    body_img = original_img[y1:y2, x1:x2]

    # Resize to BODY_SIZE
    body_img_resized = cv2.resize(body_img, BODY_SIZE, interpolation=cv2.INTER_AREA)

    # Save
    cv2.imwrite(output_path, body_img_resized)
    return True


def process_body_dataset():
    """
    Process all images in the input directory and crop bodies.
    """
    total_images = 0
    processed_images = 0
    failed_images = 0

    for emotion in EMOTIONS:
        for level in LEVELS:
            source_dir = os.path.join(INPUT_DIR, f"{emotion}_{level}")
            target_dir = os.path.join(OUTPUT_DIR, f"{emotion}_{level}")

            if not os.path.exists(source_dir):
                print(f"Warning: Source directory {source_dir} doesn't exist. Skipping.")
                continue

            image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            print(f"Processing {emotion} - {level}: {len(image_files)} images")
            total_images += len(image_files)

            for img_file in tqdm(image_files, desc=f"{emotion}_{level}"):
                img_path = os.path.join(source_dir, img_file)
                output_path = os.path.join(target_dir, img_file)

                success = detect_crop_and_resize_body(img_path, output_path)

                if success:
                    processed_images += 1
                else:
                    failed_images += 1

    print(f"\nProcessing complete!")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Failed: {failed_images}")


if __name__ == "__main__":
    process_body_dataset()