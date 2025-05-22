import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import cv2
import librosa
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from model_architecture import EmotionRecognitionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate emotion recognition model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True, help='CSV file with test data paths and labels')
    parser.add_argument('--audio_dir', type=str, help='Directory with audio files (for real-time evaluation)')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='Directory to save evaluation results')
    parser.add_argument('--real_time', action='store_true', help='Run real-time evaluation on audio files')
    
    return parser.parse_args()

def preprocess_image(image_path):
    """
    Preprocess an image for model prediction
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image array
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img / 255.0  # Normalize
    
    return np.expand_dims(img, axis=0)  # Add batch dimension

def create_spectrogram(audio_path, output_path=None, n_mels=128, hop_length=512):
    """
    Create a mel spectrogram from an audio file
    
    Args:
        audio_path: Path to the audio file
        output_path: Path to save the spectrogram image (optional)
        n_mels: Number of mel bands
        hop_length: Hop length for STFT
        
    Returns:
        Spectrogram image as numpy array
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Create a mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create figure and plot spectrogram
    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')  # Remove axis
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    
    # Save to a temporary file to convert to array
    temp_path = 'temp_spec.png'
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Read the image and resize to 224x224
    img = cv2.imread(temp_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    
    # Remove temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return img / 255.0  # Normalize

def evaluate_model(args):
    """
    Evaluate the trained model on test data
    
    Args:
        args: Command-line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    print("Loading model...")
    model = EmotionRecognitionModel()
    model.load_model(args.model_path)
    
    # Define emotion labels
    emotions = ['pain', 'shock', 'panic', 'aggression', 'confusion']
    
    if args.real_time and args.audio_dir:
        # Real-time evaluation on audio files
        evaluate_real_time(model, args.audio_dir, emotions, args.output_dir)
    else:
        # Evaluate on test dataset
        evaluate_test_data(model, args.test_data, emotions, args.output_dir)

def evaluate_test_data(model, test_data_path, emotions, output_dir):
    """
    Evaluate model on test dataset
    
    Args:
        model: Trained emotion recognition model
        test_data_path: Path to CSV file with test data
        emotions: List of emotion labels
        output_dir: Directory to save evaluation results
    """
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(test_data_path)
    
    # Predictions and ground truth
    y_true = []
    y_pred = []
    
    print("Evaluating model on test data...")
    for _, row in test_df.iterrows():
        # Preprocess image
        img = preprocess_image(row['spectrogram_path'])
        
        # Make prediction
        prediction = model.model.predict(img)
        pred_class = np.argmax(prediction)
        
        # Append to lists
        y_true.append(row['label'])
        y_pred.append(pred_class)
    
    # Calculate metrics
    print("Calculating metrics...")
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    
    # Classification report
    cls_report = classification_report(
        y_true, 
        y_pred, 
        target_names=emotions, 
        output_dict=True
    )
    
    # Convert to DataFrame for better visualization
    cls_df = pd.DataFrame(cls_report).transpose()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        xticklabels=emotions, 
        yticklabels=emotions
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Normalized Confusion Matrix (Accuracy: {accuracy:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Save classification report
    cls_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Calculate and print per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(emotions)))
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Emotion': emotions,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, 'emotion_metrics.csv'), index=False)
    
    # Plot metrics
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(emotions))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    
    plt.xlabel('Emotion')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score by Emotion')
    plt.xticks(x, emotions, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_metrics.png'))
    
    # Print summary
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("Per-class metrics:")
    print(metrics_df)
    print(f"Detailed results saved to {output_dir}")

def evaluate_real_time(model, audio_dir, emotions, output_dir):
    """
    Evaluate model on audio files in real-time
    
    Args:
        model: Trained emotion recognition model
        audio_dir: Directory with audio files
        emotions: List of emotion labels
        output_dir: Directory to save evaluation results
    """
    # Create directory for spectrograms
    spec_dir = os.path.join(output_dir, 'spectrograms')
    os.makedirs(spec_dir, exist_ok=True)
    
    # Get all audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
    
    if not audio_files:
        print("No audio files found in the specified directory.")
        return
    
    results = []
    
    print(f"Processing {len(audio_files)} audio files...")
    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file)
        
        # Create spectrogram
        spec_path = os.path.join(spec_dir, os.path.splitext(audio_file)[0] + '.png')
        img = create_spectrogram(audio_path, spec_path)
        
        # Prepare for prediction
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model.model.predict(img_batch)
        pred_class = np.argmax(prediction)
        pred_proba = prediction[0][pred_class]
        
        # Get emotion label
        emotion = emotions[pred_class]
        
        # Add to results
        results.append({
            'File': audio_file,
            'Predicted Emotion': emotion,
            'Confidence': pred_proba
        })
        
        print(f"File: {audio_file}, Predicted: {emotion}, Confidence: {pred_proba:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'real_time_predictions.csv'), index=False)
    
    # Plot distribution of predicted emotions
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Predicted Emotion', data=results_df)
    plt.title('Distribution of Predicted Emotions')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
    
    # Plot confidence scores
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Predicted Emotion', y='Confidence', data=results_df)
    plt.title('Confidence Scores by Predicted Emotion')
    plt.xlabel('Emotion')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_scores.png'))
    
    print(f"Real-time evaluation complete. Results saved to {output_dir}")

def visualize_model_activations(model, image_path, output_dir):
    """
    Visualize model activations for a given image
    
    Args:
        model: Trained emotion recognition model
        image_path: Path to the image file
        output_dir: Directory to save visualization
    """
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Get the base model (MobileNetV2)
    base_model = model.model.layers[1]
    
    # Create a model that outputs feature maps
    activation_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[layer.output for layer in base_model.layers[1:10]]  # Get activations from first few layers
    )
    
    # Get activations
    activations = activation_model.predict(img)
    
    # Plot activations
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(3, 4, 1)
    plt.title('Original Image')
    plt.imshow(img[0])
    plt.axis('off')
    
    # Plot activations for each layer
    for i, activation in enumerate(activations[:8]):  # Plot first 8 activations
        plt.subplot(3, 4, i + 2)
        plt.title(f'Layer {i+1}')
        
        # For convolutional layers, take mean across channels
        if len(activation.shape) == 4:
            activation = np.mean(activation, axis=3)
            plt.imshow(activation[0], cmap='viridis')
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_activations.png'))
    
    print(f"Model activations visualization saved to {output_dir}")

def gradient_based_cam(model, image_path, emotion_index, output_dir):
    """
    Generate Gradient-weighted Class Activation Mapping (Grad-CAM) visualization
    
    Args:
        model: Trained emotion recognition model
        image_path: Path to the image file
        emotion_index: Index of the emotion class to visualize
        output_dir: Directory to save visualization
    """
    # Preprocess image
    img = preprocess_image(image_path)
    
    # Get the base model (MobileNetV2) and last conv layer
    base_model = model.model.layers[1]  # MobileNetV2
    last_conv_layer = None
    
    # Find the last convolutional layer
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        print("No convolutional layer found in the model")
        return
    
    # Create a model that outputs both the last conv layer activation and the final output
    grad_model = tf.keras.models.Model(
        inputs=[model.model.inputs],
        outputs=[model.model.get_layer(last_conv_layer.name).output, model.model.output]
    )
    
    # Compute gradients using GradientTape
    with tf.GradientTape() as tape:
        # Cast inputs to float32
        img_float = tf.cast(img, tf.float32)
        
        # Get conv output and model predictions
        conv_output, predictions = grad_model(img_float)
        
        # Get the prediction for the target emotion class
        target_class_prediction = predictions[:, emotion_index]
    
    # Get gradients of the target class with respect to the conv output
    gradients = tape.gradient(target_class_prediction, conv_output)
    
    # Pool gradients
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map by the gradient importance
    conv_output = conv_output.numpy()[0]
    pooled_gradients = pooled_gradients.numpy()
    
    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] *= pooled_gradients[i]
    
    # Average over all channels to get the heatmap
    heatmap = np.mean(conv_output, axis=-1)
    
    # ReLU to only keep positive values
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize heatmap
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    
    # Load the original image
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (224, 224))
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    
    # Plot original image and heatmap
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(orig_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Grad-CAM for Emotion: {emotion_index}')
    plt.imshow(superimposed_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'grad_cam_emotion_{emotion_index}.png'))
    
    print(f"Grad-CAM visualization saved to {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
