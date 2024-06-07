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
        self.sequence_length = sequence_length
        self._build_model()

    def _build_model(self):
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=256, kernel_size=32),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8) 
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=64, kernel_size=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=int((self.sequence_length - 32)/8 - 4 + 1))
        )
        # generate feature

        self.mlp = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        output_feature = x
        x = x.view(-1, 64)  # Flatten the tensor
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)

    def get_embed(self, x): # used for get the feature
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.squeeze(-1)
        return x