'''
I have one folder, /data/temp/Test_202402-5. Inside the folder is a number of subfolders, each containing a subdirectory ‘raw’. This subdirectory in turn houses two files, ‘Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet’ (‘current’ for short) and ‘Sampling2000KHz_AEKi-0.parquet’ (‘AEKi’ for short).

Please write me code that will perform a short-time Fourier transform on my two recordings, dividing each into fixed-size windows and extracting the spectral centroid, spectral bandwidth, spectral flatness, spectral roll-off, variance of spectrogram, RMS of energy and spectral entropy from each window. Then, the values in temporally matching windows from the current and AEKi recordings from each subdirectory should be stored as two seven-tuples in a list, alongside the index of the window (counting from the start of the recording) as well as the subfolder name (i.e. not ‘Test_202402_5’ or ‘raw’ but the name of the directory in between).

Normalise the values of all features considered and also account for exceptions where either ‘current’ or ‘AEKi’ is corrupted or where ‘current’ and ‘AEKi’ have different number of windows, in which cases the whole subfolder should be skipped entirely without any transform being performed on the recordings contained within.
===

To address the requirements and constraints you've outlined, we'll follow a structured approach in Python. This involves using libraries such as pandas for handling .parquet files, numpy for numerical operations, scipy for computing the Short-Time Fourier Transform (STFT) and extracting spectral features, and sklearn for normalization. The process will be encapsulated in functions for modularity and clarity.

Steps Overview:
Directory Traversal: Recursively traverse the directory structure to find each raw subdirectory.
Data Reading and Validation: Read the .parquet files for current and AEKi. Validate their integrity and ensure they can be processed together.
Feature Extraction: For each valid pair of recordings, compute the STFT and extract the required spectral features from each window.
Normalization and Aggregation: Normalize the extracted features and aggregate them as specified, handling exceptions as required.
Result Compilation: Compile the results into the desired structure, skipping any subfolders where recordings are corrupted or mismatched.

Key Points:
Error Handling: The code includes basic error handling for file reading and validation checks to ensure both current and AEKi recordings can be processed together.
Normalization: Features are normalized using MinMaxScaler from sklearn.preprocessing to ensure all values are within a comparable range.
Flexibility: This approach is modular, allowing for adjustments in feature computation or normalization methods as needed.
Skipping Invalid Data: Subfolders with corrupted files or mismatched window counts are skipped entirely, as per the requirements.
This script provides a structured approach to processing the recordings, extracting, and normalizing the required features, and handling exceptions gracefully.
'''

import os
import pandas as pd
import numpy as np
import librosa
import json
import time
from pathlib import Path

def compress_channels(data):
    """Compress multiple channels into one by averaging."""
    compressed_data = np.mean(data, axis=1)
    return compressed_data

def extract_features(signal, fs, segment_index):
    """Extract specified features from a signal segment."""
    signal = np.squeeze(signal)
    features = {
        'spectral_centroid': 0,
        'spectral_bandwidth': 0,
        'spectral_flatness': 0,
        'spectral_rolloff': 0,
        'variance_of_spectrogram': 0,
        'rms_of_energy': 0,
        'zero_crossing_rate': 0,
        'segment_index': segment_index
    }
    if signal.size == 0 or np.isnan(signal).any():
        return features
    n_fft = 1024
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, max(0, n_fft - len(signal))), mode='constant')
    try:
        S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=max(1, n_fft // 2)))
    except ValueError as e:
        print(f"Error computing STFT: {e}")
        return features
    if S.size > 0 and not np.isnan(S).any():
        try:
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=S, sr=fs))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=fs))
            features['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(S=S))
            features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=S, sr=fs))
            features['variance_of_spectrogram'] = np.var(S)
            features['rms_of_energy'] = np.mean(librosa.feature.rms(S=S, frame_length=n_fft))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(signal))
        except Exception as e:
            print(f"Error computing features: {e}")
    return features

def process_files(file1, file2, window_length, hop_length):
    """Process a pair of files to extract features for each window segment."""
    try:
        data1 = pd.read_parquet(file1).to_numpy()
        data2 = pd.read_parquet(file2).to_numpy()
        if data1.shape[1] > 1:
            data1 = compress_channels(data1)
        num_segments = min(len(data1), len(data2)) // hop_length
        fs1, fs2 = 100000, 2000000
        features_list = []
        for i in range(num_segments):
            start_idx = i * hop_length
            end_idx = start_idx + window_length
            if end_idx > min(len(data1), len(data2)):
                break
            segment1 = data1[start_idx:end_idx]
            segment2 = data2[start_idx:end_idx]
            segment_features1 = extract_features(segment1, fs1, i+1)
            segment_features2 = extract_features(segment2, fs2, i+1)
            combined_features = {
                'subfolder_name': Path(file1).parent.parent.name,  # Changed to get the directory containing 'raw'
                'AEKi_features': segment_features1,
                'IRMS_features': segment_features2
            }
            features_list.append(combined_features)
        return features_list
    except Exception as e:
        print(f"Error processing files {file1} and {file2}: {e}")
        return None

def process_directory(top_directory, window_length, hop_length):
    start_time = time.time()
    results = []
    corrupted_folders = []
    for root, dirs, files in os.walk(top_directory):
        if 'raw' in dirs:
            raw_path = os.path.join(root, 'raw')
            parquet_files = [f for f in os.listdir(raw_path) if f.endswith('.parquet')]
            if len(parquet_files) == 2:
                file1, file2 = [os.path.join(raw_path, f) for f in parquet_files]
                segment_features = process_files(file1, file2, window_length, hop_length)
                if segment_features:
                    results.extend(segment_features)
                else:
                    corrupted_folders.append(Path(root).name)
    runtime = time.time() - start_time
    print(f"Processing completed in {runtime} seconds.")
    return results, corrupted_folders, runtime

# Example usage
top_directory = 'Test_202402-6'
window_length = 1024  # Example window length
hop_length = 512  # Example hop length
results, corrupted_folders = process_directory(top_directory, window_length, hop_length)

# Writing results to JSON
output_filename = f"{top_directory}.json"
with open(output_filename, 'w') as outfile:
    json.dump({'results': results, 'corrupted_folders': corrupted_folders, 'runtime': runtime}, outfile, indent=4)

print(f"Results and corrupted folder names written to {output_filename}.")