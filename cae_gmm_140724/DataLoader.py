import os
import pandas as pd

def load_training_files(directory):
    train_files = []
    for root, dirs, files in os.walk(directory):
        if '/raw' in root or '\\raw' in root:  # Adjusted for both Unix and Windows paths
            ae_file = os.path.join(root, 'Sampling2000KHz_AEKi-0.parquet')
            ec_file = os.path.join(root, 'Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet')
            if os.path.exists(ae_file) and os.path.exists(ec_file):
                train_files.append((ae_file, ec_file))
    print(f"Train files: {train_files}")
    return train_files

def load_test_files(directory):
    test_files = []
    for root, dirs, files in os.walk(directory):
        if 'raw' in dirs:  # Check if 'raw' is a direct child of the current directory
            parent_dir = os.path.basename(root)  # Get the name of the parent directory of 'raw'
            ae_file = os.path.join(root, 'raw', 'Sampling2000KHz_AEKi-0.parquet')
            ec_file = os.path.join(root, 'raw', 'Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet')
            if os.path.exists(ae_file) and os.path.exists(ec_file) and "OK" in root:
                test_files.append((parent_dir, ae_file, ec_file))
    print(f"Test files: {test_files}")
    return test_files

def load_recording(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {file_path} with shape {df.shape}")
        recording = df.iloc[:, 0].values
        return recording
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def calculate_segment_size(sampling_rate, segment_duration_ms):
    return int(sampling_rate * segment_duration_ms / 1000)
