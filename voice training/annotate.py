import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import cv2
import pandas as pd
import numpy as np

# Define paths
audio_base_dir = "C:/Users/Alex/Desktop/voice_data_balanced_augmented"
output_spec_dir = "C:/Users/Alex/Desktop/spectrograms"
csv_output_path = "C:/Users/Alex/Desktop/data.csv"

# Define emotion → label index mapping
label_map = {
    "pain": 0,
    "shock": 1,
    "panic": 2,
    "aggression": 3,
    "confusion": 4
}

def create_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(2.24, 2.24), dpi=100)  # 224x224 pixels
    librosa.display.specshow(mel_spec_db, sr=sr)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    os.makedirs(output_spec_dir, exist_ok=True)
    records = []

    for emotion_folder in os.listdir(audio_base_dir):
        emotion_path = os.path.join(audio_base_dir, emotion_folder)
        if not os.path.isdir(emotion_path):
            continue

        label = label_map.get(emotion_folder.lower())
        if label is None:
            print(f"Skipping unknown folder: {emotion_folder}")
            continue

        for file in os.listdir(emotion_path):
            if not file.lower().endswith(('.wav', '.mp3', '.ogg')):
                continue

            audio_path = os.path.join(emotion_path, file)
            spec_filename = f"{emotion_folder}_{os.path.splitext(file)[0]}.png"
            spec_path = os.path.join(output_spec_dir, spec_filename)

            create_spectrogram(audio_path, spec_path)

            # Ensure size 224x224
            img = cv2.imread(spec_path)
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(spec_path, img)

            records.append({
                "spectrogram_path": spec_path,
                "label": label
            })

    df = pd.DataFrame(records)
    df.to_csv(csv_output_path, index=False)
    print(f"✅ Done. CSV saved to: {csv_output_path}")

if __name__ == "__main__":
    main()
