import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(data_path='data/audio_speech_actors_01-24'):
    emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    data = []
    labels = []

    for actor in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor)
        if os.path.isdir(actor_path):  # Check if it's a directory
            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(actor_path, file)
                    y, sr = librosa.load(file_path, sr=None)
                    emotion = emotions[file.split('-')[2]]
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    data.append(mfcc_mean)
                    labels.append(emotion)

    data = np.array(data)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = np.eye(len(emotions))[labels]  # One-hot encoding

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
