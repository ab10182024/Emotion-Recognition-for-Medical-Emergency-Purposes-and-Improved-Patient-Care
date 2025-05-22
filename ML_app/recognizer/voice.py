### voice.py

import numpy as np
import tensorflow as tf
import librosa
import json
import cv2
import sounddevice as sd

# === CONFIGURATION ===
MODEL_PATH = 'models/voice_model.h5'
LABEL_MAPPING_PATH = 'models/label_mapping_voice.json'
SAMPLE_RATE = 16000
DURATION = 3
TARGET_SIZE = (224, 224)

# === Charger le modèle vocal MobileNetV2 ===
def load_voice_model(path=MODEL_PATH):
    try:
        model = tf.keras.models.load_model(path)
        print(f"[INFO] Modèle MobileNetV2 vocal chargé depuis {path}")
        return model
    except Exception as e:
        print(f"[ERREUR] Échec du chargement du modèle vocal : {e}")
        return None

# === Charger le mapping idx -> label ===
def load_voice_label_mapping(path=LABEL_MAPPING_PATH):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERREUR] Échec du chargement du mapping vocal : {e}")
        return {}

# === Convertir un segment audio en image MFCC compatible MobileNetV2 ===
def audio_array_to_mfcc_image(audio_array, sr=SAMPLE_RATE, target_size=TARGET_SIZE):
    try:
        mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=40)
        mfcc_norm = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
        mfcc_resized = cv2.resize(mfcc_norm, target_size)
        rgb_image = np.stack([mfcc_resized]*3, axis=-1)
        return np.expand_dims(rgb_image.astype("float32"), axis=0)
    except Exception as e:
        print(f"[ERREUR] Conversion MFCC -> image échouée : {e}")
        return None

# === Enregistrement audio depuis le micro ===
def record_voice_direct(duration=DURATION, fs=SAMPLE_RATE):
    try:
        print("[INFO] Enregistrement vocal...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print(f"[ERREUR] Échec enregistrement audio : {e}")
        return None

# === Prédiction MobileNetV2 à partir d'un segment audio ===
def predict_voice_from_array(audio_array, sr=SAMPLE_RATE, model=None, label_mapping=None):
    try:
        if audio_array is None or len(audio_array) == 0:
            print("[AVERTISSEMENT] Aucun audio capturé.")
            return "Erreur", 0.0

        image_input = audio_array_to_mfcc_image(audio_array, sr=sr)
        if image_input is None:
            print("[AVERTISSEMENT] Image audio invalide.")
            return "Erreur", 0.0

        prediction = model.predict(image_input)
        idx = int(np.argmax(prediction))

        # Mapping simple : retourne les classes principales par index
        label_mapping = {
            "0": "pain",
            "1": "shock",
            "2": "panic",
            "3": "aggression",
            "4": "confusion"
        }

        label = label_mapping.get(str(idx), f"Classe {idx}")
        confidence = float(prediction[0][idx])
        return label, confidence

    except Exception as e:
        print(f"[ERREUR] Prédiction MobileNetV2 échouée : {e}")
        return "Erreur", 0.0
