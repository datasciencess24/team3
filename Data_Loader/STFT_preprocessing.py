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
from scipy.signal import stft
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler

def read_parquet_file(filepath):
    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def compute_features(signal, fs):
    f, t, Zxx = stft(signal, fs=fs)
    magnitude = np.abs(Zxx)
    power = magnitude**2
    features = []
    for i in range(power.shape[1]):
        centroid = np.sum(f * power[:, i]) / np.sum(power[:, i])
        bandwidth = np.sqrt(np.sum((f - centroid)**2 * power[:, i]) / np.sum(power[:, i]))
        flatness = np.exp(np.mean(np.log(power[:, i]))) / np.mean(power[:, i])
        rolloff = f[np.where(np.cumsum(power[:, i]) >= np.sum(power[:, i]) * 0.85)[0][0]]
        var = np.var(power[:, i])
        rms = np.sqrt(np.mean(power[:, i]))
        spec_entropy = entropy(power[:, i] / np.sum(power[:, i]))
        features.append([centroid, bandwidth, flatness, rolloff, var, rms, spec_entropy])
    return np.array(features)

def normalize_features(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

def process_subfolder(subfolder_path):
    current_path = os.path.join(subfolder_path, 'raw', 'Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet')
    aeki_path = os.path.join(subfolder_path, 'raw', 'Sampling2000KHz_AEKi-0.parquet')
    current_signal = read_parquet_file(current_path)
    aeki_signal = read_parquet_file(aeki_path)
    if current_signal is None or aeki_signal is None:
        return None
    current_features = compute_features(current_signal.values.flatten(), 100000)
    aeki_features = compute_features(aeki_signal.values.flatten(), 2000000)
    if current_features.shape[0] != aeki_features.shape[0]:
        return None
    current_features = normalize_features(current_features)
    aeki_features = normalize_features(aeki_features)
    return current_features, aeki_features

def main(directory):
    results = []
    for root, dirs, files in os.walk(directory):
        if 'raw' in root.split(os.sep):
            subfolder_name = root.split(os.sep)[-3]
            features = process_subfolder(root)
            if features is not None:
                current_features, aeki_features = features
                for i in range(current_features.shape[0]):
                    results.append((i, subfolder_name, tuple(current_features[i]), tuple(aeki_features[i])))
    return results

# Example usage
directory = '/data/temp/Test_202402-5'
results = main(directory)