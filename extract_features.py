import os
import librosa
import numpy as np
import pandas as pd

# Path to dataset
DATA_PATH = "data/ravdess"

# Emotion labels mapping (from RAVDESS filename convention)
emotion_map = {
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
    """Extract MFCC features from an audio file."""
    signal, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # average across time
    return mfccs_mean

def get_label_from_filename(filename):
    """Parse emotion code from filename and map to label."""
    parts = filename.split("-")
    emotion_code = parts[2]  # third part is emotion
    return emotion_map.get(emotion_code, "unknown")

# Collect features and labels
features = []
labels = []

for actor in os.listdir(DATA_PATH):
    actor_path = os.path.join(DATA_PATH, actor)
    if not os.path.isdir(actor_path):
        continue
    
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            file_path = os.path.join(actor_path, file)
            mfccs = extract_features(file_path)
            label = get_label_from_filename(file)
            
            features.append(mfccs)
            labels.append(label)

# Convert to DataFrame
df = pd.DataFrame(features)
df['label'] = labels

# Save to CSV
df.to_csv("features.csv", index=False)
print("Feature extraction complete! Saved to features.csv")
