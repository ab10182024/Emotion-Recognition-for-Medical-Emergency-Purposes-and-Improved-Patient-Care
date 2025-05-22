import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

def split_audio_on_silence(file_path, save_dir, base_name, top_db=30, min_duration=0.5):
    y, sr = librosa.load(file_path, sr=None)

    # Trouver les intervalles non silencieux
    intervals = librosa.effects.split(y, top_db=top_db)

    part_count = 0
    for i, (start, end) in enumerate(intervals):
        duration = (end - start) / sr
        if duration < min_duration:
            continue  # Ignorer les segments trop courts

        segment = y[start:end]
        segment_filename = f"{base_name}_part{i+1}.wav"
        segment_path = os.path.join(save_dir, segment_filename)
        sf.write(segment_path, segment, sr)
        part_count += 1

    return part_count

def clean_and_split_dataset(input_dir="C:/Users/Alex/Desktop/voicedata", output_dir="C:/Users/Alex/Desktop/voice_data_cropped"):
    os.makedirs(output_dir, exist_ok=True)

    for emotion in os.listdir(input_dir):
        emotion_path = os.path.join(input_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
        
        output_emotion_path = os.path.join(output_dir, emotion)
        os.makedirs(output_emotion_path, exist_ok=True)

        print(f"📁 Traitement de : {emotion}")
        for filename in tqdm(os.listdir(emotion_path)):
            if filename.endswith(".mp3"):
                file_path = os.path.join(emotion_path, filename)
                base_name = os.path.splitext(filename)[0]
                split_audio_on_silence(file_path, output_emotion_path, base_name)

if __name__ == "__main__":
    clean_and_split_dataset()
    print("✅ Découpage terminé. Résultats dans `voice_data_cropped/`")
