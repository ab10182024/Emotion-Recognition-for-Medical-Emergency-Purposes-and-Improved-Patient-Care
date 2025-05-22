import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# Define paths
DATA_DIR = "C:/Users/Alex/Desktop/mixed_face_data"  # Directory containing original images
OUTPUT_DIR = "C:/Users/Alex/Desktop/augmented_face_data"  # Directory for augmented images

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create directories for each emotion and severity level
# EMOTIONS = ["Pain", "Shock", "Unconsciousness", "Confusion", "Aggression", "Panic"]
EMOTIONS = ["Unconsciousness"]
LEVELS = ["Mild", "Moderate", "Severe"]

for emotion in EMOTIONS:
    for level in LEVELS:
        os.makedirs(os.path.join(OUTPUT_DIR, f"{emotion}_{level}"), exist_ok=True)

# Define augmentation pipeline
def get_augmentation_pipeline():
    return A.Compose([
        # Spatial transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=30, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # Pixel-level transformations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        
        # Color transformations
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        
        # Advanced transformations
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        # Pixelation and other effects
        A.Downscale(scale_min=0.6, scale_max=0.9, p=0.3),
        A.CoarseDropout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.3),
    ])

# Function to load and augment images
def augment_data(num_augmentations_per_image=20):
    """
    Augment data for all emotions and severity levels
    
    Args:
        num_augmentations_per_image: Number of augmented versions to create per original image
    """
    
    # Get the augmentation pipeline
    aug_pipeline = get_augmentation_pipeline()
    
    total_original = 0
    total_augmented = 0
    
    # Process each emotion and severity level
    for emotion in EMOTIONS:
        for level in LEVELS:
            source_dir = os.path.join(DATA_DIR, f"{emotion}_{level}")
            target_dir = os.path.join(OUTPUT_DIR, f"{emotion}_{level}")
            
            # Skip if source directory doesn't exist
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory {source_dir} doesn't exist. Skipping.")
                continue
            
            # Get list of images in the source directory
            image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"Processing {emotion} - {level}: {len(image_files)} original images")
            total_original += len(image_files)
            
            # Process each image
            for img_file in tqdm(image_files, desc=f"{emotion}_{level}"):
                img_path = os.path.join(source_dir, img_file)
                
                # Read the image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read {img_path}. Skipping.")
                    continue
                
                # Convert from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Copy the original image to the target directory
                cv2.imwrite(os.path.join(target_dir, img_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # Create augmented versions
                for i in range(num_augmentations_per_image):
                    augmented = aug_pipeline(image=image)
                    aug_image = augmented['image']
                    
                    # Save augmented image
                    aug_filename = f"{os.path.splitext(img_file)[0]}_aug_{i}{os.path.splitext(img_file)[1]}"
                    cv2.imwrite(os.path.join(target_dir, aug_filename), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    total_augmented += 1
    
    print(f"Augmentation complete. Original images: {total_original}, Augmented images created: {total_augmented}")
    return total_original, total_augmented

# Function to show sample augmentations for a single image
def show_sample_augmentations(image_path, num_samples=5):
    """
    Display sample augmentations for a single image
    
    Args:
        image_path: Path to the image to augment
        num_samples: Number of augmented samples to display
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get augmentation pipeline
    aug_pipeline = get_augmentation_pipeline()
    
    # Create and display augmented versions
    plt.figure(figsize=(15, 10))
    
    # Display original image
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')
    
    # Display augmented images
    for i in range(num_samples):
        augmented = aug_pipeline(image=image)
        aug_image = augmented['image']
        
        plt.subplot(2, 3, i+2)
        plt.imshow(aug_image)
        plt.title(f"Augmented {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the augmentation
    total_original, total_augmented = augment_data(num_augmentations_per_image=20)
    
    # Show sample augmentations if a sample image exists
    sample_emotion = EMOTIONS[0]
    sample_level = LEVELS[0]
    sample_dir = os.path.join(DATA_DIR, f"{sample_emotion}_{sample_level}")
    
    if os.path.exists(sample_dir):
        sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if sample_images:
            sample_path = os.path.join(sample_dir, sample_images[0])
            show_sample_augmentations(sample_path)
    
    print("Data augmentation completed")