import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class time_series_dataset(Dataset):
    def __init__(self, data_root, files):
        # load the data
        # add the label

        self.data_root = data_root # it should have two types
        self.files = files
        # self.file_AE= 'raw/Sampling2000KHz_AEKi-0.parquet'
        # self.file_AC = 'raw/Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current ' \
        # #                'L2-Grinding spindle current L3-0.parquet'
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


    # def _load_AE_data(self):
    #     dataframe_OK_AE = []
    #     for file_name in self.folder_OK:
    #         file_path = os.path.join(self.data_root[0], file_name, self.file_AE)
    #         if os.path.isfile(file_path):
    #             df = pd.read_parquet(file_path, engine='fastparquet')
    #             print(df.head())
    #             dataframe_OK_AE.append(df)
    #     dataframe_OK_AE['label'] = 1
    #
    #     dataframe_NOK_AE = []
    #     for file_name in self.folder_NOK:
    #         file_path = os.path.join(self.data_root[1], file_name, self.file_AE)
    #         if os.path.isfile(file_path):
    #             df = pd.read_parquet(file_path, engine='fastparquet')
    #             dataframe_NOK_AE.append(df)
    #     dataframe_NOK_AE['label'] = 0
    #
    #     self.dataframe_AE = pd.concat([dataframe_OK_AE,dataframe_NOK_AE],axis=0)
    #     return self.dataframe_AE
    #
    # def _load_AC_data(self):
    #
    #     dataframe_OK_AC = []
    #     for file_name in self.folder_OK:
    #         file_path = os.path.join(self.dic_OK, file_name, self.file_AC)
    #         if os.path.isfile(file_path):
    #             df = pd.read_parquet(file_path, engine='fastparquet')
    #             dataframe_OK_AC.append(df)
    #     dataframe_OK_AC['lable'] = 1
    #
    #     dataframe_NOK_AC = []
    #     for file_name in self.folder_NOK:
    #         file_path = os.path.join(self.dic_NOK, file_name, self.file_AC)
    #         if os.path.isfile(file_path):
    #             df = pd.read_parquet(file_path, engine='fastparquet')
    #             dataframe_NOK_AC.append(df)
    #     dataframe_NOK_AC['lable'] = 0
    #
    #     self.dataframe_AC = pd.concat([dataframe_OK_AC,dataframe_NOK_AC],axis=0)
    #     return self.dataframe_AC


# # Step 4: Use DataLoader
# dataset = CustomDataset(df)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
# # Iterate through the DataLoader
# for features, labels in dataloader:
#     print("Features: ", features)
#     print("Labels: ", labels)
