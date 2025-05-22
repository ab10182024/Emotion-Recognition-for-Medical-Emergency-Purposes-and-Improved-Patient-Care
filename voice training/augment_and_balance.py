import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random
from math import ceil

# Paramètres
input_dir = "C:/Users/Alex/Desktop/voice_data_cropped"
output_dir = "C:/Users/Alex/Desktop/voice_data_balanced_augmented"
target_size = 1000  # nombre de fichiers souhaités par classe
min_duration = 0.5  # en secondes

def augment_sample(y, sr, method):
    if method == "pitch":
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
    elif method == "stretch":
        return librosa.effects.time_stretch(y, rate=0.9)
    elif method == "noise":
        noise = 0.005 * np.random.randn(len(y))
        return y + noise
    return y

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def balance_class(emotion, files):
    emotion_input_path = os.path.join(input_dir, emotion)
    emotion_output_path = os.path.join(output_dir, emotion)
    ensure_dir(emotion_output_path)

    # Copier les originaux
    for f in files:
        src = os.path.join(emotion_input_path, f)
        dst = os.path.join(emotion_output_path, f)
        sf.write(dst, *librosa.load(src, sr=None))

    current_count = len(files)
    needed = target_size - current_count

    if needed <= 0:
        return  # déjà assez de données

    print(f"🔁 Augmenting {emotion}: need {needed} more samples...")

    augmentation_methods = ["pitch", "stretch", "noise"]
    file_index = current_count + 1

    while needed > 0:
        # Choisir un fichier original
        f = random.choice(files)
        y, sr = librosa.load(os.path.join(emotion_input_path, f), sr=None)

        if librosa.get_duration(y=y, sr=sr) < min_duration:
            continue

        method = random.choice(augmentation_methods)
        y_aug = augment_sample(y, sr, method)
        aug_filename = os.path.splitext(f)[0] + f"_aug{file_index}.wav"
        sf.write(os.path.join(emotion_output_path, aug_filename), y_aug, sr)

        file_index += 1
        needed -= 1

def balance_dataset():
    for emotion in os.listdir(input_dir):
        emotion_path = os.path.join(input_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue

        files = [f for f in os.listdir(emotion_path) if f.endswith(".wav")]
        print(f"📂 {emotion}: {len(files)} fichiers")
        balance_class(emotion, files)

if __name__ == "__main__":
    balance_dataset()
    print("✅ Données équilibrées dans `voice_data_balanced/`")
