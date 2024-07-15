from DataLoader import load_recording
import numpy as np
from sklearn.preprocessing import StandardScaler

AE_SAMPLING_RATE = 2000 * 1000  # 2000 kHz
EC_SAMPLING_RATE = 100 * 1000
def load_and_segment_recordings(ae_file, ec_file, segment_duration_ms, pad_value=0):
    ae_recording = load_recording(ae_file)
    ec_recording = load_recording(ec_file)

    if ae_recording is None or ec_recording is None:
        return None, None

    segment_size_ae = calculate_segment_size(AE_SAMPLING_RATE, segment_duration_ms)
    segment_size_ec = calculate_segment_size(EC_SAMPLING_RATE, segment_duration_ms)

    num_segments_ae = len(ae_recording) // segment_size_ae
    num_segments_ec = len(ec_recording) // segment_size_ec

    ae_segments = []
    ec_segments = []

    min_length = min(len(ae_recording), len(ec_recording))

    for i in range(min(num_segments_ae, num_segments_ec)):
        ae_start_idx = i * segment_size_ae
        ae_end_idx = ae_start_idx + segment_size_ae
        ec_start_idx = i * segment_size_ec
        ec_end_idx = ec_start_idx + segment_size_ec

        # Check boundaries
        if ae_end_idx <= len(ae_recording) and ec_end_idx <= len(ec_recording):
            ae_segment = ae_recording[ae_start_idx:ae_end_idx]
            ec_segment = ec_recording[ec_start_idx:ec_end_idx]

            # Ensure segments are of equal length, discard if not aligned
            if len(ae_segment) == segment_size_ae and len(ec_segment) == segment_size_ec:
                ae_segments.append(ae_segment)
                ec_segments.append(ec_segment)

    print(f"Number of aligned segments: {len(ae_segments)}")

    return ae_segments, ec_segments


def process_and_extract_features(train_files, segment_duration_ms):
    all_features_raw = []

    for ae_file, ec_file in train_files:
        ae_segments, ec_segments = load_and_segment_recordings(ae_file, ec_file, segment_duration_ms)
        if ae_segments is None or ec_segments is None:
            continue

        for ae_segment, ec_segment in zip(ae_segments, ec_segments):
            features = calculate_features_for_pair(ae_segment, ec_segment)
            all_features_raw.append(features)

    all_features_raw = np.array(all_features_raw)

    if len(all_features_raw) == 0:
        print("Error: all_features_raw is empty. Please check data extraction.")
        return None, None, None

    # Use StandardScaler for normalisation
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features_raw)

    # Extract normalisation parameters
    feature_means = scaler.mean_
    feature_stds = scaler.scale_

    return all_features, feature_means, feature_stds


def calculate_segment_size(sampling_rate, segment_duration_ms):
    return int(sampling_rate * segment_duration_ms / 1000)

def segment_and_pad(recording, segment_size, pad_value=0):
    segments = [recording[i:i+segment_size] for i in range(0, len(recording), segment_size)]
    padded_segments = [np.pad(segment, (0, max(0, segment_size - len(segment))), 'constant', constant_values=pad_value) for segment in segments]
    return padded_segments


def calculate_features(segment, is_true):
    if is_true:
        variance = np.var(segment)
        energy = np.sum(np.square(segment))
        return variance, energy
    else:
        energy = np.sum(np.square(segment))
        return energy
def calculate_features_for_pair(ae_segment, ec_segment):
    ae_variance, ae_energy = calculate_features(ae_segment, is_true=True)
    ec_energy = calculate_features(ec_segment, is_true=False)
    return ae_variance, ae_energy, ec_energy