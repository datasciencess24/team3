# Code for baseline autoencoder model

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(True),
            nn.Linear(12, 6)
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(True),
            nn.Linear(12, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_and_evaluate_model(model, train_loader, test_loader, labels, epochs=50, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    start_time = time.time()
    for epoch in range(epochs):
        for data in train_loader:
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    training_time = time.time() - start_time
    
    # Evaluate the model
    start_time = time.time()
    reconstruction_errors = []
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            outputs = model(inputs)
            batch_errors = ((outputs - targets) ** 2).mean(axis=1)
            reconstruction_errors.extend(batch_errors.tolist())
    inference_time = time.time() - start_time

    reconstruction_errors = np.array(reconstruction_errors)
    threshold = np.percentile(reconstruction_errors, 95)
    y_pred = (reconstruction_errors > threshold).astype(int)

    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    if labels.ndim > 1:
        labels = labels.flatten()

    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)

    return model, training_time, precision, recall, inference_time
