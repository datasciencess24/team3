import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import time

def run_and_visualise_oc_svm(training_data, test_data, labels):
    # Split preprocessed training data for training and validation
    X_train, X_val = train_test_split(training_data, test_size=0.2, random_state=42)

    # Initialise One-Class SVM
    oc_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.2)  # nu=0.2 to balance between recall and false negatives

    # Measure training time
    start_time = time.time()
    oc_svm.fit(X_train)
    training_time = time.time() - start_time
    print(f"Training Time: {training_time:.4f} seconds")

    # Measure inference time
    start_time = time.time()
    predictions = oc_svm.predict(test_data)
    inference_time = time.time() - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")

    # Adjust predictions to match label format (1 for normal, 0 for abnormal)
    predictions = (predictions == 1).astype(int) * 2 - 1  # This will convert 0's to -1 and keep 1's as is
    print(predictions)

    # Calculate precision and recall
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Function to estimate feature importance
    def feature_importance(model, data):
        base_score = model.decision_function(data).mean()
        importances = []
        for i in range(data.shape[1]):
            # Perturb feature i
            save = data[:, i].copy()
            np.random.shuffle(data[:, i])
            
            # Measure change in the decision function
            new_score = model.decision_function(data).mean()
            data[:, i] = save  # Restore original data
            
            # A larger change means the feature is more important
            importances.append(abs(new_score - base_score))
        
        return np.array(importances)

    # Calculate relative feature importance
    importances = feature_importance(oc_svm, X_val.copy())
    print("Feature importances:", importances)

    # Visualise relative feature importance
    features = range(test_data.shape[1])
    plt.bar(features, importances)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(features)
    plt.title('Feature Importance')
    plt.show()