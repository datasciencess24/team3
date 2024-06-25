import tuple_loader
import gmm_for_tuples

training_data = tuple_loader.load_training_data(grinding_ok_train)
test_data, labels = tuple_loader.load_test_data(grinding_test)

# Scale data
scaler = StandardScaler()
training_data_scaled = scaler.fit_transform(training_data)
test_data_scaled = scaler.transform(test_data)

gmm_model, threshold_score = gmm_for_tuples.train_gmm(training_data)
modified_test_data = gmm_for_tuples.process_test_data(test_data, gmm_model, threshold_score)
precision, recall = gmm_for_tuples.evaluate_predictions(modified_test_data, labels)
print(f"Precision: {precision}, Recall: {recall}")