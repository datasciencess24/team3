import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

class TimeSeriesCNN(nn.Module):
    def __init__(self, config):
        super(TimeSeriesCNN, self).__init__()
        self.config = config
        self._build_model()

    def _build_model(self):
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=32),  # 使用的是1维卷积层
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8)  # 用于降维 注意区分max pooling（最大值池化）和卷积核的操作区别：池化作用于图像中不重合的区域（这与卷积操作不同）
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=int((self.size - 32) / 8) - 4 + 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(64 * (self.sequence_length // 2), 256),
            nn.ReLU(),
            nn.nn.Linear(256, self.config['num_features'])
        )
        self.fc1 = nn.Linear(self.config['num_features'], 2)
        # self.fc1 = nn.Linear(256, 2)  # Adjust based on your input size
        # # self.fc2 = nn.Linear(256, self.config['num_features'])  # Number of features you want to generate

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 64 * (self.sequence_length // 2))  # Flatten the tensor
        x = x.mlp(x)
        feature_output = x
        x = self.fc1(x)
        return x, feature_output