import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

def train_isolation_forest(X_train, test_data_scaled, y_test):
    # Start timing for training
    training_start_time = time.time()
    
    # Initialize and train the Isolation Forest model
    model = IsolationForest(n_estimators=50, contamination='auto', random_state=42)
    model.fit(X_train)
    
    # End timing for training
    training_end_time = time.time()
    
    # Measure inference time
    inference_start_time = time.time()
    # Predict anomalies on the test set. Isolation Forest labels anomalies as -1, so convert to 1 for consistency
    y_pred = model.predict(test_data_scaled)
    y_pred = np.where(y_pred == 1, 0, 1)  # Convert -1 to 1 for anomalies, and 1 to 0 for normal points
    
    inference_end_time = time.time()
    
    # Calculate precision and recall
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Print results
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Training time: {training_end_time - training_start_time:.2f} seconds")
    print(f"Inference time: {inference_end_time - inference_start_time:.2f} seconds")
    
    # For explainability: Feature importances are not directly available from Isolation Forest.
    # However, you can look at the average path lengths in the trees for insights.
    # Shorter paths to isolation can indicate more anomalous instances.
    
    return model

