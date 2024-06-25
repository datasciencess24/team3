import numpy as np
import json

def load_training_data(filename):
    with open(filename, 'r') as file:
        training_data = json.load(file)
    # Convert list of tuples to a NumPy array for easier processing later
    return np.array(training_data) # To change

def load_test_data(filename):
    with open(filename, 'r') as file:
        test_data_with_labels = json.load(file)
    # Separate features and labels
    test_data = np.array([t[:-1] for t in test_data_with_labels])
    labels = np.array([t[-1] for t in test_data_with_labels])
    return test_data, labels