import torch
import os
import argparse
from torch.nn.utils.rnn import pad_sequence
import json
import logging
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import heapq
import numpy as np
import torch

from Models.CNN import TimeSeriesCNN



def train_timeseries_Conv(train_dataset, device, args, sequence_length): 
    # Hyper parameters
    batch_size = args.train_batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    loss_list = []

    model = TimeSeriesCNN(args, device, sequence_length).to(device)  
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    # Train the model
    total_step = len(train_dataset)  
    for epoch in range(num_epochs):  
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_dataset): 
            if (i + 1) % 2 == 0:  # 99，199，299，399...
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)  
                predicted = outputs.max(dim=1)[1]
                total += labels.size(0)  
                correct += (predicted == labels).sum().item()  

                print('Valid Accuracy of the model on the valid dataset: {} %'.format(100 * correct / total))
            else:
                images,labels = images.to(device),labels.to(device)
                labels = labels.long()
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels) 
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_value = loss.item()
            loss_list.append(loss_value)
    # sava the model
    model_path = '../cache/'
    if (os.path.exists(model_path) == False):
        os.makedirs(model_path)
    torch.save(model, model_path + 'Timeseries_Conv_model_1.pkl')
    return model
