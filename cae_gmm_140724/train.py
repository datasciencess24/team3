import torch
from Model import ConvAutoencoder
import time
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
import joblib
import datetime
from data_preprocessing import process_and_extract_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def extract_features_and_train(train_files, segment_duration_ms):
    autoencoder = ConvAutoencoder().to(device)
    start_time = time.time()  # Record start time

    # Process and extract features
    all_features, feature_means, feature_stds = process_and_extract_features(train_files, segment_duration_ms)

    if all_features is None:
        return None, None, None, None

    features_tensor = torch.tensor(all_features, dtype=torch.float32).unsqueeze(1)
    features_tensor = features_tensor.to(device)

    if features_tensor.shape[0] == 0 or features_tensor.shape[1] != 1 or features_tensor.shape[2] != 3:
        print("Error: features_tensor is incorrectly shaped.")
        return None, None, None, None

    print("Shape of features_tensor:", features_tensor.shape)

    criterion = nn.MSELoss()
    optimiser = optim.Adam(autoencoder.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        optimiser.zero_grad()
        outputs = autoencoder(features_tensor)
        print(f"Epoch {epoch + 1}, Input shape: {features_tensor.shape}, Output shape: {outputs.shape}")
        loss = criterion(outputs, features_tensor)
        loss.backward()
        optimiser.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    encoded_features = autoencoder.encoder(features_tensor).detach()

    encoded_features_2d = encoded_features.cpu().view(encoded_features.size(0), -1).numpy()

    # Gaussian Mixture Model defined here
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(encoded_features_2d)

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y%m%d_%H%M%S')

    # Save trained model and normalisation parameters
    # Replace with desired output path
    model_path = f'/output_path/cae_gmm_model_{formatted_time}.pkl''
    joblib.dump((autoencoder, gmm, feature_means, feature_stds), model_path)
    print(f"Model saved to {model_path}")

    end_time = time.time()  # Record end time
    print(f"Training time: {end_time - start_time} seconds")  # Print training duration

    return autoencoder, gmm, feature_means, feature_stds


