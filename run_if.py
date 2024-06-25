# Run Isolation Forest model

import tuple_loader
import if_for_tuples

# Load data
training_data = tuple_loader.load_training_data(grinding_ok_train)
test_data, labels = tuple_loader.load_test_data(grinding_test)

# Scale data
scaler = StandardScaler()
training_data_scaled = scaler.fit_transform(training_data)
test_data_scaled = scaler.transform(test_data)

# Train the model and measure performance
model = if_for_tuples.train_isolation_forest(training_data_scaled, test_data_scaled, labels)