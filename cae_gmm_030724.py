import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import joblib
import scipy.stats as stats
import time
import json

# At the beginning of the script, ensure PyTorch is using CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
AE_SAMPLING_RATE = 2000 * 1000  # 2000 kHz
EC_SAMPLING_RATE = 100 * 1000   # 100 kHz
SEGMENT_DURATION_MS = 50  # Segment length in milliseconds
ANOMALY_THRESHOLD = np.log(0.025)  # Threshold for segment anomaly (likelihood less than 2.5% = anomalous)
RECORDING_ANOMALY_THRESHOLD = 99.99  # Percentile threshold for recording anomaly (more anomalous segments than 99.99% of recordings = anomalous; does not use log likelihood!)

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

def segment_and_pad(recording, segment_size, pad_value=0):
    segments = [recording[i:i+segment_size] for i in range(0, len(recording), segment_size)]
    padded_segments = [np.pad(segment, (0, max(0, segment_size - len(segment))), 'constant', constant_values=pad_value) for segment in segments]
    return padded_segments

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

def calculate_features(segment):
    variance = np.var(segment)
    energy = np.sum(np.square(segment))
    return variance, energy

def calculate_features_for_pair(ae_segment, ec_segment):
    ae_variance, ae_energy = calculate_features(ae_segment)
    ec_variance, ec_energy = calculate_features(ec_segment)
    return ae_variance, ae_energy, ec_variance, ec_energy

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=2, padding=1).to(device),
            nn.ReLU().to(device),
            nn.Conv1d(16, 4, 3, stride=2, padding=1).to(device),
            nn.ReLU().to(device),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 16, 3, stride=2, padding=1, output_padding=1).to(device),
            nn.ReLU().to(device),
            nn.ConvTranspose1d(16, 1, 3, stride=2, padding=1, output_padding=1).to(device),
            nn.Sigmoid().to(device),
        )

    def forward(self, x):
        x = x.to(device)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
        return None, None

    # Use StandardScaler for normalisation
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features_raw)

    # Extract normalisation parameters
    feature_means = scaler.mean_
    feature_stds = scaler.scale_

    return all_features, feature_means, feature_stds

def extract_features_and_train(train_files, segment_duration_ms):
    autoencoder = ConvAutoencoder().to(device)
    start_time = time.time()  # Record start time

    # Process and extract features
    all_features, feature_means, feature_stds = process_and_extract_features(train_files, segment_duration_ms)

    if all_features is None:
        return None, None, None, None, None

    features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(1)
    features_tensor = features_tensor.to(device)

    if features_tensor.shape[0] == 0 or features_tensor.shape[1] != 1:
        print("Error: features_tensor is incorrectly shaped.")
        return None, None, None, None, None

    print("Shape of features_tensor:", features_tensor.shape)

    criterion = nn.MSELoss()
    optimiser = optim.Adam(autoencoder.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        optimiser.zero_grad()
        outputs = autoencoder(features_tensor)
        loss = criterion(outputs, features_tensor)
        loss.backward()
        optimiser.step()

    encoded_features = autoencoder.encoder(features_tensor).detach()

    encoded_features_2d = encoded_features.cpu().view(encoded_features.size(0), -1).numpy()

    # Gaussian Mixture Model defined here
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(encoded_features_2d)

    # Save trained model and normalisation parameters
    model_path = '/home/dsbwl24_team003/data/temp/Test_202402-6/cae_gmm_model.pkl'
    joblib.dump((autoencoder, gmm, feature_means, feature_stds), model_path)

    end_time = time.time()  # Record end time
    print(f"Training time: {end_time - start_time} seconds")  # Print training duration

    return autoencoder, gmm, feature_means, feature_stds

def detect_anomalies(autoencoder, gmm, ae_file, ec_file, segment_duration_ms, feature_means, feature_stds):
    ae_segments, ec_segments = load_and_segment_recordings(ae_file, ec_file, segment_duration_ms)
    if ae_segments is None or ec_segments is None:
        return None, None

    all_features = []

    start_time = time.time()  # Record start time for inference

    for ae_segment, ec_segment in zip(ae_segments, ec_segments):
        features = calculate_features_for_pair(ae_segment, ec_segment)
        all_features.append(features)

    all_features = np.array(all_features)

    end_time = time.time()  # Record end time for inference
    print(f"Inference time: {end_time - start_time} seconds")  # Print inference duration

    if len(all_features) == 0:
        print("Error: all_features is empty. Please check data extraction.")
        return None, None

    # Normalize features using the saved means and stds
    all_features = (all_features - feature_means) / feature_stds

    features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        features_tensor = features_tensor.to(device)
        encoded_features = autoencoder.encoder(features_tensor).detach()
        encoded_features_2d = encoded_features.cpu().view(encoded_features.size(0), -1).numpy()

    # Identify anomalous segments using GMM scores
    log_likelihoods = gmm.score_samples(encoded_features_2d)

    # Debug statement to check the log-likelihoods
    # print('Log-likelihoods:', log_likelihoods)

    # Obtain negative log-likelihoods; the greater the value, the more anomalous
    neg_log_likelihoods = -(log_likelihoods)

    # Calculate the proportion of anomalous segments
    proportion_anomalous = np.sum(log_likelihoods < ANOMALY_THRESHOLD) / len(log_likelihoods)

    print('Proportion anomalous:', proportion_anomalous)
    return proportion_anomalous, neg_log_likelihoods

def flag_recording_as_anomalous(proportion_anomalous, normal_proportions):
    percentile = stats.percentileofscore(normal_proportions, proportion_anomalous)
    if percentile > RECORDING_ANOMALY_THRESHOLD:
        percentile = 100
    is_anomalous = percentile > RECORDING_ANOMALY_THRESHOLD
    return is_anomalous, percentile

def save_to_json(data, output_file):
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f'Test results saved to {output_file}')

# Define the directory containing your training data
training_data_directory = "/home/dsbwl24_team003/data/temp/Test_202402-6"

# Define the desired segment duration in milliseconds
segment_duration_ms = 50

# Load training and testing file paths
train_files = load_training_files(training_data_directory)

# Train the model and get normalisation parameters
autoencoder, gmm, feature_means, feature_stds = extract_features_and_train(train_files, segment_duration_ms)


# Load NOK test files and flag recordings as anomalous if needed
test_data_directory = "/home/dsbwl24_team003/data/temp/NOK_Measurements"

test_files = load_test_files(test_data_directory)
results = {}

for root, ae_file, ec_file in test_files:
    print(f"Processing files in {root}")

    # Load and segment recordings
    ae_segments, ec_segments = load_and_segment_recordings(ae_file, ec_file, segment_duration_ms)
    if ae_segments is None or ec_segments is None:
        print(f"Error loading files in {root}. Skipping...")
        continue

    all_features = []

    # Process each segment
    for ae_segment, ec_segment in zip(ae_segments, ec_segments):
        # Calculate features for the segment pair
        features = calculate_features_for_pair(ae_segment, ec_segment)
        all_features.append(features)

    all_features = np.array(all_features)

    if len(all_features) == 0:
        print(f"No valid segments found in {root}. Skipping...")
        continue

    # Normalise features using training parameters
    all_features_normalised = (all_features - feature_means) / feature_stds

    # Convert to tensor
    features_tensor = torch.tensor(all_features_normalised, dtype=torch.float32).unsqueeze(1).to(device)

    # Run inference through autoencoder and GMM
    with torch.no_grad():
        encoded_features = autoencoder.encoder(features_tensor).detach()
        encoded_features_2d = encoded_features.cpu().view(encoded_features.size(0), -1).numpy()

    # Score with GMM and detect anomalies
    scores = gmm.score_samples(encoded_features_2d)
    proportion_anomalous, neg_log_likelihoods = detect_anomalies(autoencoder, gmm, ae_file, ec_file, segment_duration_ms, feature_means, feature_stds)

    # Determine if recording is anomalous and its percentile
    is_anomalous, percentile = flag_recording_as_anomalous(proportion_anomalous, normal_proportions)

    results[root] = {
        'status': 'anomalous' if is_anomalous else 'normal',
        'proportion_anomalous': proportion_anomalous,
        'percentile': percentile,
        'neg_log_likelihoods': neg_log_likelihoods.tolist()  # Convert numpy array to list for JSON serialization
    }

# Save results to JSON file
output_file = f'{test_data_directory}/anomaly_results(NOK_Measurements 030724).json'
save_to_json(results, output_file)


# Load OK test files to check for false positive rate
test_data_directory = "/home/dsbwl24_team003/data/temp/Test_202402-5"

test_files = load_test_files(test_data_directory)
results = {}

for root, ae_file, ec_file in test_files:
    print(f"Processing files in {root}")

    # Load and segment recordings
    ae_segments, ec_segments = load_and_segment_recordings(ae_file, ec_file, segment_duration_ms)
    if ae_segments is None or ec_segments is None:
        print(f"Error loading files in {root}. Skipping...")
        continue

    all_features = []

    # Process each segment
    for ae_segment, ec_segment in zip(ae_segments, ec_segments):
        # Calculate features for the segment pair
        features = calculate_features_for_pair(ae_segment, ec_segment)
        all_features.append(features)

    all_features = np.array(all_features)

    if len(all_features) == 0:
        print(f"No valid segments found in {root}. Skipping...")
        continue

    # Normalise features using training parameters
    all_features_normalised = (all_features - feature_means) / feature_stds

    # Convert to tensor
    features_tensor = torch.tensor(all_features_normalised, dtype=torch.float32).unsqueeze(1).to(device)

    # Run inference through autoencoder and GMM
    with torch.no_grad():
        encoded_features = autoencoder.encoder(features_tensor).detach()
        encoded_features_2d = encoded_features.cpu().view(encoded_features.size(0), -1).numpy()

    # Score with GMM and detect anomalies
    scores = gmm.score_samples(encoded_features_2d)
    proportion_anomalous, neg_log_likelihoods = detect_anomalies(autoencoder, gmm, ae_file, ec_file, segment_duration_ms, feature_means, feature_stds)

    # Determine if recording is anomalous and its percentile
    is_anomalous, percentile = flag_recording_as_anomalous(proportion_anomalous, normal_proportions)

    results[root] = {
        'status': 'anomalous' if is_anomalous else 'normal',
        'proportion_anomalous': proportion_anomalous,
        'percentile': percentile,
        'neg_log_likelihoods': neg_log_likelihoods.tolist()  # Convert numpy array to list for JSON serialization
    }

# Save results to JSON file
output_file = f'{test_data_directory}/anomaly_results(OK_Measurements 030724).json'
save_to_json(results, output_file)
