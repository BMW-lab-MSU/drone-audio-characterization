# Script for extracting audio features from a CNN for drone classification
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
""" import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense """

# Loop through all `.wav` files in the directory
def loadWav(data_dir, tArray):
    # Get all .wav files and sort by modification time (oldest first)
    wav_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith('.wav')],
        key=lambda f: os.path.getmtime(os.path.join(data_dir, f))
    )

    for filename in wav_files:
        file_path = os.path.join(data_dir, filename)

        # Load audio file while preserving original sampling rate
        audio, sampling_rate = librosa.load(file_path, sr=None)

        # Append audio data
        tArray.append(audio)

    return tArray

# Load all '.wav' files in the Blackmore directory (all audio recordings are recorded at 44100 Hz)
sr = 44100
data_dir = r'B:\drone-audio\2024-12-14'
recordings = []
recordings = loadWav(data_dir,recordings)
recordings = np.array(recordings)

# Normalize recordings
for i in range(len(recordings)):
    signal = recordings[i] 
    max_val = np.max(np.abs(signal))
    if max_val != 0:  # Avoid division by zero
        recordings[i] = signal / max_val  # Normalize signal and reassign

# Mel-Frequnecy Spectrogram of each audio signal
# with 20 & 50 ms frames
# 50% overlap, and a hanning window
nfft_points_50 = int(.05 * sr)
step_size_50 = int(nfft_points_50 * 0.5)
nfft_points_20 = int(.02 * sr)
step_size_20 = int(nfft_points_20 * 0.5)

melspectrograms_50 = []
melspectrograms_20 = []

# 50 ms frames
for i in range(len(recordings)):
    # Calculate spectrogram using Librosa
    mel_spec_50 = librosa.feature.melspectrogram(y=recordings[i], sr=sr, n_fft=nfft_points_50, hop_length=step_size_50, window='hann')
    
    # Convert to log scale (dB)
    mel_spec_50_db = librosa.power_to_db(mel_spec_50, ref=np.max)

    # Append
    melspectrograms_50.append(mel_spec_50_db)

# 20 ms frames
for i in range(len(recordings)):
    # Calculate spectrogram using Librosa
    mel_spec_20 = librosa.feature.melspectrogram(y=recordings[i], sr=sr, n_fft=nfft_points_20, hop_length=step_size_20, window='hann')
    
    # Convert to log scale (dB)
    mel_spec_20_db = librosa.power_to_db(mel_spec_20, ref=np.max)

    # Append
    melspectrograms_20.append(mel_spec_20_db)

# Convert to numpy arrays
melspectrograms_50 = np.array(melspectrograms_50)
melspectrograms_20 = np.array(melspectrograms_20)
