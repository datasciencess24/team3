import torch
import torch.nn as nn
import numpy as np
import time
import scipy.stats as stats
from data_preprocessing import load_and_segment_recordings,calculate_features_for_pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEGMENT_DURATION_MS = 50  # Segment length in milliseconds
ANOMALY_THRESHOLD = np.log(0.025)
RECORDING_ANOMALY_THRESHOLD = 99.99

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=1, padding=1).to(device),
            nn.ReLU().to(device),
            nn.Conv1d(16, 8, 3, stride=1, padding=1).to(device),
            nn.ReLU().to(device),
            nn.Conv1d(8, 3, 3, stride=1, padding=1).to(device),
            nn.ReLU().to(device),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(3, 8, 3, stride=1, padding=1).to(device),
            nn.ReLU().to(device),
            nn.Conv1d(8, 16, 3, stride=1, padding=1).to(device),
            nn.ReLU().to(device),
            nn.Conv1d(16, 1, 3, stride=1, padding=1).to(device),
            nn.Sigmoid().to(device),
        )

    def forward(self, x):
        x = x.to(device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



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

    # Normalise features using the saved means and stds
    all_features = (all_features - feature_means) / feature_stds

    features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        features_tensor = features_tensor.to(device)
        encoded_features = autoencoder.encoder(features_tensor).detach()
        encoded_features_2d = encoded_features.cpu().view(encoded_features.size(0), -1).numpy()

    # Identify anomalous segments using GMM scores
    log_likelihoods = gmm.score_samples(encoded_features_2d)

    # Obtain negative log-likelihoods; the greater the value, the more anomalous
    neg_log_likelihoods = -(log_likelihoods)

    # Calculate the proportion of anomalous segments
    proportion_anomalous = np.sum(log_likelihoods < ANOMALY_THRESHOLD) / len(log_likelihoods)

    print('Proportion anomalous:', proportion_anomalous * 100, '%')
    print('Negative log-likelihoods:', neg_log_likelihoods)
    return proportion_anomalous, neg_log_likelihoods


def flag_recording_as_anomalous(proportion_anomalous, normal_proportions):
    percentile = stats.percentileofscore(normal_proportions, proportion_anomalous)
    if percentile > RECORDING_ANOMALY_THRESHOLD:
        percentile = 100
    is_anomalous = percentile > RECORDING_ANOMALY_THRESHOLD
    return is_anomalous, percentile

