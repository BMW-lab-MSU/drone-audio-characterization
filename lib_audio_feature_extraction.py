# Script for extracting audio features using Librosa for drone classification
import os
import numpy as np
import librosa

# Loop through all `.wav` files in the directory
def loadWav(data_dir, tArray):
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(data_dir, filename)
            
            # Load audio file using librosa
            audio, sampling_rate = librosa.load(file_path, sr=None)  # sr=None preserves original sampling rate
            
            # Append audio data and sampling rate
            tArray.append((audio))
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

# 20 & 40 ms windows with 50% overlap
nfft_points_50 = int(.05 * sr)
step_size_50 = int(nfft_points_50 * 0.5)
nfft_points_20 = int(.02 * sr)
step_size_20 = int(nfft_points_20 * 0.5)

# Lists to store features for each window size
features_20ms = []
features_50ms = []

# Extract features for each recording (Hanning window)
for i, signal in enumerate(recordings):
    # 20 ms window features
    mfcc_20 = librosa.feature.mfcc(y=signal, sr=sr, n_fft=nfft_points_20, hop_length=step_size_20, n_mfcc=13, window='hann')
    delta_mfcc_20 = librosa.feature.delta(mfcc_20)
    spectral_centroid_20 = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=nfft_points_20, hop_length=step_size_20, window='hann')
    spectral_bandwidth_20 = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=nfft_points_20, hop_length=step_size_20, window='hann')
    spectral_contrast_20 = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=nfft_points_20, hop_length=step_size_20, window='hann')
    spectral_flatness_20 = librosa.feature.spectral_flatness(y=signal, n_fft=nfft_points_20, hop_length=step_size_20, window='hann')
    rms_energy_20 = librosa.feature.rms(y=signal, frame_length=nfft_points_20, hop_length=step_size_20)
    zcr_20 = librosa.feature.zero_crossing_rate(y=signal, frame_length=nfft_points_20, hop_length=step_size_20)

    # Append and average features across all frames
    curfeatures_20ms = np.hstack([
        np.mean(mfcc_20, axis=1), 
        np.std(mfcc_20, axis=1),
        np.mean(delta_mfcc_20, axis=1), 
        np.std(delta_mfcc_20, axis=1),
        np.mean(spectral_centroid_20, axis=1), 
        np.std(spectral_centroid_20, axis=1),
        np.mean(spectral_bandwidth_20, axis=1), 
        np.std(spectral_bandwidth_20, axis=1),
        np.mean(spectral_contrast_20, axis=1), 
        np.std(spectral_contrast_20, axis=1),
        np.mean(spectral_flatness_20, axis=1), 
        np.std(spectral_flatness_20, axis=1),
        np.mean(rms_energy_20, axis=1), 
        np.std(rms_energy_20, axis=1),
        np.mean(zcr_20, axis=1), 
        np.std(zcr_20, axis=1)
    ])

    features_20ms.append(curfeatures_20ms)

    # 50 ms window features
    mfcc_50 = librosa.feature.mfcc(y=signal, sr=sr, n_fft=nfft_points_50, hop_length=step_size_50, n_mfcc=13, window='hann')
    delta_mfcc_50 = librosa.feature.delta(mfcc_50)
    spectral_centroid_50 = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=nfft_points_50, hop_length=step_size_50, window='hann')
    spectral_bandwidth_50 = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=nfft_points_50, hop_length=step_size_50, window='hann')
    spectral_contrast_50 = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=nfft_points_50, hop_length=step_size_50, window='hann')
    spectral_flatness_50 = librosa.feature.spectral_flatness(y=signal, n_fft=nfft_points_50, hop_length=step_size_50, window='hann')
    rms_energy_50 = librosa.feature.rms(y=signal, frame_length=nfft_points_50, hop_length=step_size_50)
    zcr_50 = librosa.feature.zero_crossing_rate(y=signal, frame_length=nfft_points_50, hop_length=step_size_50)

    curfeatures_50ms = np.hstack([
        np.mean(mfcc_50, axis=1), 
        np.std(mfcc_50, axis=1),
        np.mean(delta_mfcc_50, axis=1), 
        np.std(delta_mfcc_50, axis=1),
        np.mean(spectral_centroid_50, axis=1), 
        np.std(spectral_centroid_50, axis=1),
        np.mean(spectral_bandwidth_50, axis=1), 
        np.std(spectral_bandwidth_50, axis=1),
        np.mean(spectral_contrast_50, axis=1), 
        np.std(spectral_contrast_50, axis=1),
        np.mean(spectral_flatness_50, axis=1), 
        np.std(spectral_flatness_50, axis=1),
        np.mean(rms_energy_50, axis=1), 
        np.std(rms_energy_50, axis=1),
        np.mean(zcr_50, axis=1), 
        np.std(zcr_50, axis=1)
    ])

    features_50ms.append(curfeatures_50ms)

# Convert lists to NumPy arrays
features_20ms = np.array(features_20ms)
features_50ms = np.array(features_50ms)

print(features_20ms.shape)
print(features_50ms.shape)

np.save("features_20ms.npy", features_20ms)
np.save("features_50ms.npy", features_50ms)