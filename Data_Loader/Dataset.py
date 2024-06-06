import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class time_series_dataset(Dataset):
    def __init__(self, data_root, files):
        self.data_root = data_root # it should have two types
        self.files = files
        self.folder_OK = os.listdir(self.data_root[0]).remove('.DS_Store')
        self.folder_NOK = os.listdir(self.data_root[1]).remove('.DS_Store')
        self.dataframe = self._load_data()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        features = torch.tensor(row[:-1].values, dtype=torch.float32)
        label = torch.tensor(row[-1], dtype=torch.long)
        return features, label

