# Runs baseline autoencoder model
import tuple_loader
import ae_for_tuples

# Load data
training_data = tuple_loader.load_training_data(grinding_ok_train)
test_data, labels = tuple_loader.load_test_data(grinding_test)

# Scale data
scaler = StandardScaler()
training_data_scaled = scaler.fit_transform(training_data)
test_data_scaled = scaler.transform(test_data)

# Convert test data into PyTorch tensors
training_data_scaled = torch.FloatTensor(training_data_scaled) 
test_data_scaled = torch.FloatTensor(test_data_scaled)
labels = torch.LongTensor(labels)  # Assuming labels are for classification and are integers

# Create DataLoader instances
train_dataset = TensorDataset(training_data_scaled, training_data_scaled) 
test_dataset = TensorDataset(test_data_scaled, test_data_scaled)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = ae_for_tuples.Autoencoder()
model, training_time, precision, recall, inference_time = train_and_evaluate_model(model, train_loader, test_loader, labels)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
print(f"Training time: {training_time:.2f} seconds, Inference time: {inference_time:.2f} seconds")