import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.utils.data as data_utils
import torch
from torch.nn.utils.rnn import pad_sequence
from .Dataset import time_series_dataset

from copy import deepcopy


class DataModule:
    """
    Base class for all data loaders
    """

    def __init__(self, data_root, args):
        self.data_root = data_root
        self.folder = os.listdir(self.data_root)
        self.files = ['raw/Sampling2000KHz_AEKi-0.parquet', 'raw/Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet']
        self.args = args


    def _load_data(self):
        ########### for loading the data##############
        #Step 1: load the data from 2 files and keep them in one data frame
        #Step 2: loop add all the data in two one data frame.
        dataset = []
        for file_name in self.folder:
            file_path = os.path.join(self.data_root, file_name)
            print(file_path)
            features = pd.DataFrame()
            for file in self.files:
                f = os.path.join(file_path , file)
                print(f)
                feature = pd.read_parquet(f, engine='fastparquet').iloc[:50]
                features = pd.concat([features, feature], axis=1)
            dataset.append(features)  # the length is how many data we have
            self.dataset = dataset
        return dataset

def collate_fn(batch):
    tensors, targets = zip(*batch) # unzip the data
    features = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    return features, targets