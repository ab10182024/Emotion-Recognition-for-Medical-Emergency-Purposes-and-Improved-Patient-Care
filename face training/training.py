import os
import numpy as np
import tensorflow as tf
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from model_architecture import (
    EmotionDataGenerator, 
    train_model, 
    plot_training_history, 
    get_class_weights,
    IMAGE_SIZE, 
    BATCH_SIZE
)

EMOTIONS = ["Pain", "Shock", "Unconsciousness", "Confusion", "Aggression", "Panic"]
LEVELS = ["Mild", "Moderate", "Severe"]

# Set memory growth for GPU to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# Function to split data into train/val/test sets
def split_data(augmented_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split the augmented data into train, validation, and test sets
    
    Args:
        augmented_dir: Directory containing augmented data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
    
    Returns:
        Dictionary containing file paths and labels for each split
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    # Lists to store file paths and labels
    all_files = []
    all_labels = []
    
    # Label mapping dictionary
    label_mapping = {}
    label_idx = 0
    
    # Collect all file paths and labels
    for emotion in EMOTIONS:
        for level in LEVELS:
            class_name = f"{emotion}_{level}"
            class_dir = os.path.join(augmented_dir, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            # Assign a numeric label to this class
            if class_name not in label_mapping:
                label_mapping[class_name] = label_idx
                label_idx += 1
            
            # Collect files for this class
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            all_files.extend(files)
            all_labels.extend([label_mapping[class_name]] * len(files))
    
    # Split into train and temporary set
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=(val_ratio + test_ratio), stratify=all_labels, random_state=42
    )
    
    # Split temporary set into validation and test sets
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=(1 - val_ratio_adjusted), stratify=temp_labels, random_state=42
    )
    
    # Create and return the data split dictionary
    data_splits = {
        'train': {'files': train_files, 'labels': train_labels},
        'val': {'files': val_files, 'labels': val_labels},
        'test': {'files': test_files, 'labels': test_labels},
        'label_mapping': label_mapping,
        'idx_to_class': {v: k for k, v in label_mapping.items()}
    }
    
    # Print statistics
    print("Data split statistics:")
    print(f"  Training set: {len(train_files)} images")
    print(f"  Validation set: {len(val_files)} images")
    print(f"  Test set: {len(test_files)} images")
    print(f"  Total: {len(all_files)} images")
    print(f"  Number of classes: {len(label_mapping)}")
    
    return data_splits

# Main training function
def main():
    """
    Main function to load data and train the model
    """
    # Load the data splits
    try:
        with open('C:/Users/Alex/Desktop/data_splits.json', 'r') as f:
            data_splits = json.load(f)
            
        # Convert lists to numpy arrays
        for split in ['train', 'val', 'test']:
            data_splits[split]['labels'] = np.array(data_splits[split]['labels'])
    except FileNotFoundError:
        # If data splits don't exist, try importing from data_augmentation script
        try:
            OUTPUT_DIR = "C:/Users/Alex/Desktop/resized_face_data"
            data_splits = split_data(OUTPUT_DIR)
            
            # Save data splits for future use
            with open('C:/Users/Alex/Desktop/data_splits.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                data_splits_json = {
                    'train': {
                        'files': data_splits['train']['files'],
                        'labels': data_splits['train']['labels']
                    },
                    'val': {
                        'files': data_splits['val']['files'],
                        'labels': data_splits['val']['labels']
                    },
                    'test': {
                        'files': data_splits['test']['files'],
                        'labels': data_splits['test']['labels']
                    },
                    'label_mapping': data_splits['label_mapping'],
                    'idx_to_class': data_splits['idx_to_class']
                }

                json.dump(data_splits_json, f)
        except ImportError:
            print("Error: Could not find data splits or data_augmentation script.")
            return
    
    # Create data generators
    train_generator = EmotionDataGenerator(
        files=data_splits['train']['files'],
        labels=data_splits['train']['labels'],
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        augment=True  # Enable additional augmentation during training
    )
    
    val_generator = EmotionDataGenerator(
        files=data_splits['val']['files'],
        labels=data_splits['val']['labels'],
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
        augment=False
    )
    
    # Calculate class weights to handle imbalanced data
    class_weights = get_class_weights(data_splits['train']['labels'])
    print("Class weights:", class_weights)
    
    # Train the model
    model, history = train_model(train_generator, val_generator, class_weights)
    
    # Plot training history
    plot_training_history(history)
    
    # Save label mapping for inference
    with open('C:/Users/Alex/Desktop/label_mapping.json', 'w') as f:
        json.dump({
            'label_mapping': data_splits['label_mapping'],
            'idx_to_class': data_splits['idx_to_class']
        }, f)
    
    print("Training complete! Model saved as 'model_final.h5'")
    print("Label mapping saved as 'label_mapping.json'")

if __name__ == "__main__":
    main()