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
        # self.file_AE = 'raw/Sampling2000KHz_AEKi-0.parquet'
        # self.file_AC = 'raw/Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current ' \
        #                'L2-Grinding spindle current L3-0.parquet'
        # Get two data frame
        # self.AE_dataframe = time_series_dataset(self.data_root,self.file_AE)._load_data()
        # self.AC_dataframe = time_series_dataset(self.data_root, self.file_AC)._load_data()


    # def df_to_tensor(df): # input dataframe return the tensor vector with the same length by using the zero padding
    #     target = df['Label'].values
    #     features = df.drop('Label', axis=1).values #feature from two different data set and need to combine
    #
    #     train = data_utils.TensorDataset(features, target)
    #     train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)


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
                # dataframe.fillna(0)
            #df = pd.DataFrame(features)
            dataset.append(features)  # the length is how many data we have
            self.dataset = dataset
        return dataset

def collate_fn(batch):
    #def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor]]):
    tensors, targets = zip(*batch) # unzip the data
    features = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    return features, targets
    # def _split_data(self, split): ## split_rate
    #     if split == 0.0:
    #         return self.dataset, None
    #
    #     if isinstance(split, int):
    #         assert split > 0
    #         assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
    #         len_valid = split
    #     else:
    #         len_valid = int(self.n_samples * split)
    #
    #     len_train = self.n_samples - len_valid
    #     train_mask = list(range(len_train))
    #     train_dataset = Subset(self.dataset, train_mask)
    #     eval_mask = list(range(len_train, len_train + len_valid))
    #     eval_dataset = Subset(self.dataset, eval_mask)
    #
    #     self.n_samples = len(train_dataset)
    #
    #     return train_dataset, eval_dataset


    # def _load_data(self): # load the current and the signal into one data frame
    #     dataframe_OK = pd.DataFrame()
    #     for file_name in self.folder_OK:
    #         file_path = os.path.join(self.data_root[0], file_name,  self.files)
    #         if os.path.isfile(file_path):
    #             df = pd.DataFrame(pd.read_parquet(file_path, engine='fastparquet'))
    #             df.columns = [file_name]
    #             dataframe_OK = pd.concat([dataframe_OK, df], axis=1)
    #     dataframe_OK['label'] = 1
    #
    #     dataframe_NOK = pd.DataFrame()
    #     for file_name in self.folder_OK:
    #         file_path = os.path.join(self.data_root[0], file_name,  self.files)
    #         if os.path.isfile(file_path):
    #             df = pd.DataFrame(pd.read_parquet(file_path, engine='fastparquet'))
    #             df.columns = [file_name] #modify to differnt column name
    #             dataframe_NOK = pd.concat([dataframe_NOK, df], axis=1)
    #     dataframe_NOK['label'] = 0
    #
    #     self.dataframe = pd.concat([dataframe_OK, dataframe_NOK],axis=0)
    #     self.dataframe.fillna(0)
    #     return self.dataframe

