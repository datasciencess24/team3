import numpy as np
import os
import json
from DataLoader import load_training_files, load_test_files, load_recording
from data_preprocessing import load_and_segment_recordings
from train import extract_features_and_train
from Model import detect_anomalies, flag_recording_as_anomalous, calculate_features_for_pair, ConvAutoencoder
import torch
import joblib
from pathlib import Path

# At the beginning of the script, ensure PyTorch is using CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_to_json(data, output_file):
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f'Test results saved to {output_file}')

def load_model(model_path, device):
    if not Path(model_path).is_file():
        print(f"No pre-trained model found at {model_path}")
        return None, None, None, None

    autoencoder, gmm, feature_means, feature_stds = joblib.load(model_path)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    
    print(f"Model loaded from {model_path}")
    return autoencoder, gmm, feature_means, feature_stds

if __name__ == '__main__':
    # Define paths
    training_data_directory = ""
    test_data_directory = ""

    # Define the desired segment duration in milliseconds
    segment_duration_ms = 50

    # Ask user if they want to train a new model or use a pre-trained one
    use_pretrained = input("Do you want to use a pre-trained model? (y/n): ").lower() == 'y'

    if use_pretrained:
        # Ask user for the path to the pre-trained model
        model_path = input("Enter the path to the pre-trained model: ").strip()
        autoencoder, gmm, feature_means, feature_stds = load_model(model_path, device)
        if autoencoder is None:
            print("Pre-trained model not found or failed to load. Training a new model.")
            use_pretrained = False

    if not use_pretrained:
        # Load training file paths
        train_files = load_training_files(training_data_directory)

        # Train the model and get normalisation parameters
        autoencoder, gmm, feature_means, feature_stds = extract_features_and_train(train_files, segment_duration_ms)

    # Load normal proportions from training data
    normal_proportions = []
    train_files = load_training_files(training_data_directory)
    for ae_file, ec_file in train_files:
        proportion_anomalous, _ = detect_anomalies(autoencoder, gmm, ae_file, ec_file, segment_duration_ms, feature_means, feature_stds)
        if proportion_anomalous is not None:
            normal_proportions.append(proportion_anomalous)

    print('Normal proportions', normal_proportions)

    # Load OK test files and flag recordings as anomalous if needed
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
            'proportion_anomalous_in_percent': proportion_anomalous * 100,
            'percentile': percentile,
            'neg_log_likelihoods': neg_log_likelihoods.tolist()  # Convert numpy array to list for JSON serialisation
        }

    # Save results to JSON file
    output_file = f'{test_data_directory}/anomaly_results.json'
    save_to_json(results, output_file)