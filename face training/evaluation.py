import os
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from model_architecture import EmotionDataGenerator, IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES
from tensorflow.keras.models import load_model

def get_class_name(idx, idx_to_class):
    return idx_to_class.get(str(idx), f"Unknown-{idx}")

def evaluate_model(model, test_generator, idx_to_class):
    test_images = []
    test_labels = []
    
    for i in range(len(test_generator)):
        batch_images, batch_labels = test_generator[i]
        test_images.append(batch_images)
        test_labels.append(batch_labels)
    
    test_images = np.vstack(test_images)
    test_labels = np.vstack(test_labels)
    
    predictions = model.predict(test_images)
    true_classes = np.argmax(test_labels, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)

    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=[get_class_name(i, idx_to_class) for i in range(NUM_CLASSES)],
        output_dict=True
    )
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    return {
        'report': report,
        'confusion_matrix': cm,
        'true_classes': true_classes,
        'predicted_classes': predicted_classes,
        'predictions': predictions,
        'test_images': test_images
    }

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('C:/Users/Alex/Desktop/face/evaluation/confusion_matrix.png')
    plt.show()

def show_misclassified_examples(test_images, true_classes, predicted_classes, idx_to_class, n=10):
    misclassified = np.where(true_classes != predicted_classes)[0]
    if len(misclassified) == 0:
        print("No misclassified examples found.")
        return
    indices = np.random.choice(misclassified, min(n, len(misclassified)), replace=False)
    
    plt.figure(figsize=(15, 12))
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)
        img = test_images[idx] * 255
        img = img.astype(np.uint8)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        true_class = get_class_name(true_classes[idx], idx_to_class)
        pred_class = get_class_name(predicted_classes[idx], idx_to_class)
        plt.title(f"True: {true_class}\nPred: {pred_class}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('C:/Users/Alex/Desktop/face/evaluation/misclassified_examples.png')
    plt.show()

def main():
    try:
        model = load_model('C:/Users/Alex/Desktop/face/model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Chemin vers toutes les images classées
    data_dir = 'C:/Users/Alex/Desktop/face/original_face_images'

    all_files = []
    all_labels = []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(class_path, fname))
                all_labels.append(class_name)

    # Créer les mappings classe <-> index
    unique_classes = sorted(set(all_labels))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    idx_to_class = {str(i): cls for cls, i in class_to_idx.items()}
    class_names = [get_class_name(i, idx_to_class) for i in range(NUM_CLASSES)]

    # Sauvegarder le mapping
    with open('C:/Users/Alex/Desktop/face/label_mapping.json', 'w') as f:
        json.dump({'idx_to_class': idx_to_class}, f)

    # Convertir les labels en entiers
    all_labels_idx = [class_to_idx[label] for label in all_labels]

    # Split automatique 80% train / 20% test
    _, test_files, _, test_labels = train_test_split(
        all_files, all_labels_idx, test_size=0.2, stratify=all_labels_idx, random_state=42
    )

    test_generator = EmotionDataGenerator(test_files, test_labels, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)

    # Évaluation
    eval_results = evaluate_model(model, test_generator, idx_to_class)
    report = eval_results['report']

    print("\nModel Performance:")
    for class_name in class_names:
        print(f"{class_name}: F1-Score = {report[class_name]['f1-score']:.4f}")
    print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")

    plot_confusion_matrix(eval_results['confusion_matrix'], class_names)
    show_misclassified_examples(
        eval_results['test_images'],
        eval_results['true_classes'],
        eval_results['predicted_classes'],
        idx_to_class
    )

    # Export CSV
    results_df = pd.DataFrame({
        'Class': class_names + ['weighted avg'],
        'Precision': [report[c]['precision'] for c in class_names] + [report['weighted avg']['precision']],
        'Recall': [report[c]['recall'] for c in class_names] + [report['weighted avg']['recall']],
        'F1-Score': [report[c]['f1-score'] for c in class_names] + [report['weighted avg']['f1-score']],
        'Support': [report[c]['support'] for c in class_names] + [report['weighted avg']['support']]
    })

    results_df.to_csv('C:/Users/Alex/Desktop/face/evaluation/evaluation_results.csv', index=False)
    print("\nEvaluation results saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
