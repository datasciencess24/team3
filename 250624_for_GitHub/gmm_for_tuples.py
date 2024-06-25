# Gaussian Mixture Model

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
from scipy.stats import scoreatpercentile
from sklearn.preprocessing import StandardScaler
import time

def train_gmm(training_data):
    kf = KFold(n_splits=10)
    top_percentiles = []
    
    training_start_time = time.time()  # Start timing for training
    
    for train_index, test_index in kf.split(training_data):
        train_data = training_data[train_index]
        test_data = training_data[test_index]
        
        gmm = GaussianMixture(n_components=1, covariance_type='full')
        gmm.fit(train_data)  # Training the model
        
        scores = -gmm.score_samples(test_data)  # Inference
        top_percentiles.append(scoreatpercentile(scores, 97.5))  # Anomaly scores at the x-th (here 97.5th) percentile taken as threshold
    
    threshold_score = np.mean(top_percentiles)
    
    # Final model training on full dataset
    final_model_start_time = time.time()
    gmm_model = GaussianMixture(n_components=1, covariance_type='full').fit(training_data)
    training_end_time = time.time()  # End timing for training
    
    # Total training time including final model training
    total_training_time = training_end_time - training_start_time
    final_model_training_time = training_end_time - final_model_start_time
    
    print(f"Total training time: {total_training_time:.2f} seconds")
    print(f"Final model training time: {final_model_training_time:.2f} seconds")
    
    # Measure inference time
    inference_start_time = time.time()
    _ = gmm_model.score_samples(training_data)  # Using training data for inference time measurement
    inference_end_time = time.time()
    
    inference_time = inference_end_time - inference_start_time
    print(f"Inference time on training data: {inference_time:.2f} seconds")
    
    return gmm_model, threshold_score

def process_test_data(test_data, gmm_model, threshold_score): # Threshold derived from percentile score in train_gmm() method
    scores = -gmm_model.score_samples(test_data)
    predictions = (scores > threshold_score).astype(int)
    return np.column_stack((test_data, scores, predictions))

def evaluate_predictions(modified_test_data, labels):
    predictions = modified_test_data[:, -1]
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    print(predictions)
    print(precision)
    return precision, recall
