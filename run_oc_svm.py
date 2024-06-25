from tuple_loader import load_training_data, load_test_data
from sklearn.preprocessing import StandardScaler
from oc_svm_for_tuples import run_and_visualise_oc_svm

# Load data
training_data = load_training_data(grinding_ok_train)
test_data, labels = load_test_data(grinding_test)
test_data, labels = test_data, [-1 if label == 1 else 1 for label in labels]

# Preprocess data
scaler = StandardScaler()
training_data_scaled = scaler.fit_transform(training_data)
test_data_scaled = scaler.transform(test_data)

# Run and visualise OC-SVM
run_and_visualise_oc_svm(training_data_scaled, test_data_scaled, labels)