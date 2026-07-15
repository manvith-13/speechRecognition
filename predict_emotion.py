import torch
import torch.nn as nn
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load label encoder (reuse from training)
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
labels = list(emotion_map.values())
encoder = LabelEncoder()
encoder.fit(labels)

# Same MFCC extraction as training
def extract_mfcc(file_path, n_mfcc=40, max_len=200):
    signal, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T   # shape (time_steps, n_mfcc)

# Define same LSTM model
class EmotionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(EmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Load trained model
input_dim = 40
hidden_dim = 128
output_dim = len(labels)
model = EmotionLSTM(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("models/emotion_lstm.pth"))
model.eval()

def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        outputs = model(mfcc_tensor)
        _, predicted = torch.max(outputs, 1)
        emotion = encoder.inverse_transform([predicted.item()])[0]
    return emotion

# Example usage
file_path = "my_audio.wav.mp3"
print("Predicted Emotion:", predict_emotion(file_path))

