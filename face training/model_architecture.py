import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras.utils import Sequence

# Create TensorBoard callback
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 18  # 6 emotions x 3 levels
LEARNING_RATE = 0.0001

# Emotion and level names for reference
EMOTIONS = ["Pain", "Shock", "Unconsciousness", "Confusion", "Aggression", "Panic"]
LEVELS = ["Mild", "Moderate", "Severe"]

# Custom data generator that loads and preprocesses images
class EmotionDataGenerator(Sequence):
    def __init__(self, files, labels, batch_size=32, image_size=(224, 224), shuffle=True, augment=False):
        self.files = files
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.n = len(files)
        self.indexes = np.arange(self.n)
        self.on_epoch_end()
        
        # Create augmentation for training if needed
        if self.augment:
            self.train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        
        # Preprocessing function from EfficientNetB0
        self.preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(self.n / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Find list of IDs
        batch_files = [self.files[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        
        # Generate data
        X, y = self._data_generation(batch_files, batch_labels)
        
        return X, y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _data_generation(self, batch_files, batch_labels):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((len(batch_files), *self.image_size, 3))
        y = np.array(batch_labels)
        
        # Generate data
        for i, file_path in enumerate(batch_files):
            # Load and preprocess image
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            
            # Apply augmentation if enabled (only for training)
            if self.augment:
                img = self.train_datagen.random_transform(img)
            
            # Preprocess image for EfficientNet
            img = self.preprocess_input(img)
            
            # Store sample
            X[i,] = img
        
        # Convert labels to one-hot encoding
        y = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)
        
        return X, y

# Build the model using EfficientNetB0 as base
def build_model(input_shape=(224, 224, 3), num_classes=18):
    """
    Build and compile the model using EfficientNetB0 as base
    
    Args:
        input_shape: Image input shape
        num_classes: Number of output classes
        
    Returns:
        Compiled model
    """
    # Create input layer
    inputs = Input(shape=input_shape)
    
    # Load pre-trained EfficientNetB0 model without top layers
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Final output layer with softmax activation
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to unfreeze layers for fine-tuning
def unfreeze_model(model, num_layers_to_unfreeze=30):
    """
    Unfreeze the last n layers of the base model for fine-tuning
    
    Args:
        model: The model to unfreeze layers in
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
        
    Returns:
        Model with unfrozen layers
    """
    # Find the base model layers
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    
    if base_model is None:
        print("Base model not found!")
        return model
    
    # Get total number of layers
    total_layers = len(base_model.layers)
    
    # Make sure we don't try to unfreeze more layers than exist
    num_layers_to_unfreeze = min(num_layers_to_unfreeze, total_layers)
    
    # Unfreeze the last n layers
    for layer in base_model.layers[total_layers - num_layers_to_unfreeze:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to calculate class weights to handle class imbalance
def get_class_weights(labels):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        labels: List of integer labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    return dict(zip(classes, class_weights))

# Function to create and train the model
def train_model(train_generator, val_generator, class_weights=None):
    """
    Create, train and save the model
    
    Args:
        train_generator: Generator for training data
        val_generator: Generator for validation data
        class_weights: Optional class weights to handle imbalance
        
    Returns:
        Trained model and training history
    """
    # Create model
    model = build_model(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
    print(model.summary())
    
    # Create callbacks
    checkpoint = ModelCheckpoint(
        'emotion_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    
    # Train the model with frozen base layers
    print("Training with frozen base model...")
    history1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Fine-tune the model by unfreezing some layers
    print("Fine-tuning the model...")
    model = unfreeze_model(model, num_layers_to_unfreeze=30)
    
    # Continue training with unfrozen layers
    history2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=40,
        callbacks=callbacks,
        class_weight=class_weights,
        initial_epoch=len(history1.history['loss'])
    )
    
    # Combine histories
    combined_history = {
        'loss': history1.history['loss'] + history2.history['loss'],
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
    }
    
    # Save the final model
    model.save('C:/Users/Alex/Desktop/model_final.h5')
    
    return model, combined_history

# Function to plot training history
def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Training history
    """
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()