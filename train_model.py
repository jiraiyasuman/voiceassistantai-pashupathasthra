import speech_recognition as sr
import librosa
import numpy as np
import pyttsx3
import soundfile
import joblib
import tempfile
import os

# Load pre-trained emotion model (you need to train & save beforehand or download a pretrained one)
MODEL_PATH = "emotion_model.pkl"  # Example saved model path
model = joblib.load(MODEL_PATH)

# Text-to-speech engine
engine = pyttsx3.init()

# Function to extract features from audio
def extract_features(audio_path):
    y, sr_rate = librosa.load(audio_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=40).T, axis=0)
    return mfccs

# Capture voice from microphone
def record_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_file.name, "wb") as f:
            f.write(audio_data.get_wav_data())
        return temp_file.name

# Predict emotion from recorded audio
def predict_emotion(audio_path):
    features = extract_features(audio_path).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

# Main program
if __name__ == "__main__":
    audio_file = record_voice()
    emotion = predict_emotion(audio_file)
    print(f"Detected Emotion: {emotion}")
    engine.say(f"I think you are feeling {emotion}")
    engine.runAndWait()
    os.remove(audio_file)
