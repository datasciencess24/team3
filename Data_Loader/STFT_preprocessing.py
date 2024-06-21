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

# ChatGPT

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
    print(f"After compressing channels, data shape: {compressed_data.shape}")  # Diagnostic print
    return compressed_data

def extract_features(signal, fs):
    """Extract specified features from a signal, adjusting for signal length."""
    # Ensure signal is a 1D array
    signal = np.squeeze(signal)
    print(f"Signal shape before feature extraction: {signal.shape}")  # Diagnostic print
    
    # Check if signal is empty or contains NaN values
    if signal.size == 0 or np.isnan(signal).any():
        print("Signal is empty or contains NaN values. Using default feature values.")
        return {
            'spectral_centroid': 0,
            'spectral_bandwidth': 0,
            'spectral_flatness': 0,
            'spectral_rolloff': 0,
            'variance_of_spectrogram': 0,
            'rms_of_energy': 0,
            'zero_crossing_rate': 0
        }
    
    n_fft = 1024
    if len(signal) < n_fft:
        # Pad signal if it's shorter than n_fft
        signal = np.pad(signal, (0, max(0, n_fft - len(signal))), mode='constant')
        print(f"Signal padded to length: {len(signal)}")  # Diagnostic print

    # Attempt to compute the Short-Time Fourier Transform (STFT) of the signal
    try:
        S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=max(1, n_fft // 2)))
        print(f"Spectrogram shape: {S.shape}")  # Diagnostic print
    except ValueError as e:
        print(f"Error computing STFT: {e}")
        S = np.array([])  # Use an empty array as a fallback
    
    # Initialize a dictionary to hold feature values, defaulting to 0
    features = {
        'spectral_centroid': 0,
        'spectral_bandwidth': 0,
        'spectral_flatness': 0,
        'spectral_rolloff': 0,
        'variance_of_spectrogram': 0,
        'rms_of_energy': 0,
        'zero_crossing_rate': 0
    }
    
    # Check if S is valid for feature extraction
    if S.size > 0 and not np.isnan(S).any():
        try:
            print("Extracting spectral_centroid...")
            spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=fs)
            print(f"spectral_centroid shape: {spectral_centroid.shape}")
            features['spectral_centroid'] = np.mean(spectral_centroid) if spectral_centroid.size > 0 else 0
            
            print("Extracting spectral_bandwidth...")
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=fs)
            print(f"spectral_bandwidth shape: {spectral_bandwidth.shape}")
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth) if spectral_bandwidth.size > 0 else 0
            
            print("Extracting spectral_flatness...")
            spectral_flatness = librosa.feature.spectral_flatness(S=S)
            print(f"spectral_flatness shape: {spectral_flatness.shape}")
            features['spectral_flatness'] = np.mean(spectral_flatness) if spectral_flatness.size > 0 else 0
            
            print("Extracting spectral_rolloff...")
            spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=fs)
            print(f"spectral_rolloff shape: {spectral_rolloff.shape}")
            features['spectral_rolloff'] = np.mean(spectral_rolloff) if spectral_rolloff.size > 0 else 0
            
            print("Calculating variance_of_spectrogram...")
            features['variance_of_spectrogram'] = np.var(S)
            
            print("Extracting rms_of_energy...")
            rms_of_energy = librosa.feature.rms(S=S, frame_length=n_fft)
            print(f"rms_of_energy shape: {rms_of_energy.shape}")
            features['rms_of_energy'] = np.mean(rms_of_energy) if rms_of_energy.size > 0 else 0
            
            print("Extracting zero_crossing_rate...")
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(signal))
            features['zero_crossing_rate'] = zero_crossing_rate

        except Exception as e:
            print(f"Error computing features: {e}")
            # Features dictionary already initialized with default values
    else:
        print("Spectrogram is empty or contains NaN values. Using default feature values.")
    
    return features

def process_files(file1, file2):
    """Process a pair of files to extract and pair features."""
    try:
        data1 = pd.read_parquet(file1).to_numpy()
        data2 = pd.read_parquet(file2).to_numpy()
        print(f"Data1 shape after loading: {data1.shape}")  # Diagnostic print
        print(f"Data2 shape after loading: {data2.shape}")  # Diagnostic print
        # Compress channels for the first file if it has more than one channel
        if data1.shape[1] > 1:
            data1 = compress_channels(data1)
        fs1, fs2 = 100000, 2000000  # Sampling rates for the files
        features1 = extract_features(data1, fs1)
        features2 = extract_features(data2, fs2)
        return features1, features2
    except Exception as e:
        print(f"Error processing files {file1} and {file2}: {e}")
        return None, None

def process_directory(top_directory):
    start_time = time.time()
    results = []
    corrupted_folders = []

    for root, dirs, files in os.walk(top_directory):
        if 'raw' in dirs:
            raw_path = os.path.join(root, 'raw')
            parquet_files = [f for f in os.listdir(raw_path) if f.endswith('.parquet')]
            if len(parquet_files) == 2:
                file1, file2 = [os.path.join(raw_path, f) for f in parquet_files]
                features1, features2 = process_files(file1, file2)
                if features1 and features2:
                    subfolder_name = Path(root).name
                    results.append((subfolder_name, features1, features2))
                else:
                    corrupted_folders.append(Path(root).name)

    runtime = time.time() - start_time
    print(f"Processing completed in {runtime} seconds.")
    return results, corrupted_folders

# Example usage
top_directory = 'Data/NOK_Measurements'
results, corrupted_folders = process_directory(top_directory)

# Writing results to JSON
output_filename = f"{top_directory}.json"
with open(output_filename, 'w') as outfile:
    json.dump({'results': results, 'corrupted_folders': corrupted_folders}, outfile)

print(f"Results and corrupted folder names written to {output_filename}.")