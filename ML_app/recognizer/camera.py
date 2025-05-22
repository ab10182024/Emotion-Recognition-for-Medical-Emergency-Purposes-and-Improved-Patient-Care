### camera.py

import threading
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import json
import time
from recognizer.voice import (
    load_voice_model,
    load_voice_label_mapping,
    record_voice_direct,
    predict_voice_from_array
)

MODEL_PATH = 'models/model_final.h5'
LABEL_MAPPING_PATH = 'models/label_mapping.json'
CONFIDENCE_THRESHOLD = 0.5

class VideoCamera:
    CRITICAL_FACE_STATES = ["Unconsciousness_Severe", "Pain_Severe", "Shock_Severe","Confusion_Severe","Aggression_Severe"]

    def __init__(self):
        self.alert_message = ""
        self.cap = cv2.VideoCapture(0)
        self.model = load_model(MODEL_PATH)
        self.label_mapping = self.load_label_mapping(LABEL_MAPPING_PATH)

        self.predicted_class = ""
        self.confidence = 0
        self.last_inference_time = 0
        self.fps_counter = 0
        self.start_time = time.time()
        self.fps = 0

        self.voice_model = load_voice_model()
        self.voice_mapping = load_voice_label_mapping()
        self.voice_pred = ""
        self.voice_conf = 0.0

        self.voice_thread = threading.Thread(target=self.run_voice_loop, daemon=True)
        self.voice_thread.start()

    def run_voice_loop(self):
        while True:
            audio_array = record_voice_direct()
            if audio_array is not None:
                pred, conf = predict_voice_from_array(
                    audio_array,
                    sr=16000,
                    model=self.voice_model,
                    label_mapping=self.voice_mapping
                )
                self.voice_pred = pred
                self.voice_conf = conf
            time.sleep(3.0)

    def __del__(self):
        self.cap.release()

    def load_label_mapping(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return {"idx_to_class": {}}

    def preprocess_frame(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None

        display_frame = frame.copy()
        current_time = time.time()

        if current_time - self.last_inference_time > 0.2:
            processed_img = self.preprocess_frame(frame)
            predictions = self.model.predict(processed_img)

            if predictions.shape[1] > 1:
                idx = int(np.argmax(predictions[0]))
                self.confidence = float(predictions[0][idx])
                self.predicted_class = self.label_mapping["idx_to_class"].get(str(idx), f"Class {idx}")
            else:
                self.predicted_class = "Valeur brute"
                self.confidence = float(predictions[0][0])

            self.last_inference_time = current_time

            self.fps_counter += 1
            if current_time - self.start_time >= 1.0:
                self.fps = self.fps_counter / (current_time - self.start_time)
                self.fps_counter = 0
                self.start_time = current_time

        # Détection des états critiques uniquement pour la prédiction faciale
        if self.predicted_class in self.CRITICAL_FACE_STATES :#and self.confidence > 0.01:
            self.alert_message = f"ETAT CRITIQUE (FACE): {self.predicted_class.upper()}"
        else:
            self.alert_message = ""

        # Fusion des prédictions faciale et vocale
        if self.predicted_class == self.voice_pred:
            final_pred = self.predicted_class
            final_conf = (self.confidence + self.voice_conf) / 2
        else:
            final_pred = self.predicted_class if self.confidence > self.voice_conf else self.voice_pred
            final_conf = max(self.confidence, self.voice_conf)

        # Affichage sur la vidéo
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.confidence > CONFIDENCE_THRESHOLD:
            cv2.putText(display_frame, f"Face: {self.predicted_class} ({self.confidence:.2f})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)

        if self.voice_conf > CONFIDENCE_THRESHOLD:
            cv2.putText(display_frame, f"Voice: {self.voice_pred} ({self.voice_conf:.2f})", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)

        if final_conf > CONFIDENCE_THRESHOLD:
            cv2.putText(display_frame, f"Fusion: {final_pred} ({final_conf:.2f})", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 200), 2)

        if self.alert_message:
            cv2.putText(display_frame, self.alert_message, (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', display_frame)
        return jpeg.tobytes()
    def get_current_alert():
        return VideoCamera().alert_message
