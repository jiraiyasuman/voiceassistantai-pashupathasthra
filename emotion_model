import os
import librosa
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Path to your dataset (e.g., RAVDESS, CREMA-D)
DATASET_PATH = "dataset/"  # Change this to your dataset folder

# List of emotions you want to classify
emotions_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

features = []
labels = []

print("🔄 Loading dataset and extracting features...")

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            if emotion_code in emotions_map:
                file_path = os.path.join(root, file)
                mfccs = extract_features(file_path)
                features.append(mfccs)
                labels.append(emotions_map[emotion_code])

features = np.array(features)
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM classifier
print("🚀 Training model...")
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc*100:.2f}%")

# Save the model
joblib.dump(model, "emotion_model.pkl")
print("💾 Model saved as emotion_model.pkl")
