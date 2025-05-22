# ✅ Updated training_code.py that skips spectrogram creation
# Assumes spectrograms are already generated and dataset CSV is available

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    parser.add_argument('--annotations_file', type=str, required=True, help='CSV file with spectrogram paths and numeric labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save models and plots')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--fine_tune_epochs', type=int, default=20)
    return parser.parse_args()

class EmotionRecognitionModel:
    def __init__(self, num_classes=5, input_shape=(224, 224, 3), learning_rate=0.0001):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        base_model = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def unfreeze_top_layers(self, num_layers=23):
        base_model = self.model.layers[1]
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate / 10), loss='categorical_crossentropy', metrics=['accuracy'])

    def save_model(self, filepath):
        self.model.save(filepath)

def get_data_generators(dataset_df, batch_size=25, test_size=0.2, val_size=0.1):
    train_df, test_df = train_test_split(dataset_df, test_size=test_size, stratify=dataset_df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['label'], random_state=42)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    def generate_from_dataframe(df, datagen):
        while True:
            df = df.sample(frac=1).reset_index(drop=True)
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                batch_x = []
                batch_y = []
                for _, row in batch_df.iterrows():
                    img = cv2.imread(row['filename'])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0
                    if datagen != test_datagen:
                        img = datagen.random_transform(img)
                    batch_x.append(img)
                    label = np.zeros(5)
                    label[int(row['label'])] = 1
                    batch_y.append(label)
                yield np.array(batch_x), np.array(batch_y)

    train_generator = generate_from_dataframe(train_df, train_datagen)
    val_generator = generate_from_dataframe(val_df, test_datagen)
    test_generator = generate_from_dataframe(test_df, test_datagen)

    train_steps = int(np.ceil(len(train_df) / batch_size))
    val_steps = int(np.ceil(len(val_df) / batch_size))
    test_steps = int(np.ceil(len(test_df) / batch_size))

    return train_generator, val_generator, test_generator, train_steps, val_steps, test_steps, train_df

def train_model(args):
    print("Loading dataset...")
    dataset_df = pd.read_csv(args.annotations_file)

    print("Creating data generators...")
    train_generator, val_generator, test_generator, train_steps, val_steps, test_steps, train_df = \
        get_data_generators(dataset_df, batch_size=args.batch_size)

    print("Building model...")
    model = EmotionRecognitionModel(num_classes=5, learning_rate=args.learning_rate)

    checkpoint_path = "C:/Users/Alex/Desktop/best_model.h5"
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    print("Training model...")
    history = model.model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    print("Fine-tuning model...")
    model.unfreeze_top_layers(num_layers=23)
    fine_tune_history = model.model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=args.fine_tune_epochs,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    all_epochs = list(range(1, args.epochs + 1)) + list(range(args.epochs + 1, args.epochs + args.fine_tune_epochs + 1))
    all_acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
    all_val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
    all_loss = history.history['loss'] + fine_tune_history.history['loss']
    all_val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(all_epochs, all_acc, label='Training Accuracy')
    plt.plot(all_epochs, all_val_acc, label='Validation Accuracy')
    plt.axvline(x=args.epochs, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(all_epochs, all_loss, label='Training Loss')
    plt.plot(all_epochs, all_val_loss, label='Validation Loss')
    plt.axvline(x=args.epochs, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("C:/Users/Alex/Desktop/training_history.png")
    model.save_model("C:/Users/Alex/Desktop/final_voice_model.h5")
    print("✅ Training complete.")

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
