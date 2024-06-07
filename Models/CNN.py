import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

class TimeSeriesCNN(nn.Module):
    def __init__(self, args, device,sequence_length):
        super(TimeSeriesCNN, self).__init__()
        self.args = args
        self.device = device
        #self.size = args.num_features # num
        self.sequence_length = sequence_length
        self._build_model()

    def _build_model(self):
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=40, kernel_size=32),  # 使用的是1维卷积层 （L-32+1）
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8)  #(L-32+1)-8+1-4+1
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=40, kernel_size=4),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=int((self.sequence_length - 32)/8 - 4 + 1))
        )
        # generate feature

        self.mlp = nn.Sequential(
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        #self.fc1 = nn.Linear(self.config['num_features'], 2)
        # self.fc1 = nn.Linear(256, 2)  # Adjust based on your input size
        # # self.fc2 = nn.Linear(256, self.config['num_features'])  # Number of features you want to generate

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        output_feature = x
        #print(output_feature)
        print(output_feature.shape) #(batch_size, channel size, length)
        x = x.view(-1, 40)  # Flatten the tensor
        print(x.shape) #(batch_size, channel size)
        x = self.mlp(x)
        print(x.shape) #(3,2) (batch_size,num_classes)
        #x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    def get_embed(self, x): # used for get the feature
        x = x.to(self.device)
        # x = x.view(-1, self.size).long()
        x = self.conv1(x)
        x = self.conv2(x)
        #x = x.squeeze(0)
        x = x.squeeze(-1) #(batch_size, feature size)
        return x