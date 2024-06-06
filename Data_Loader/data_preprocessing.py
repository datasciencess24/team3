from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.utils.data as data_utils
from sklearn.preprocessing import StandardScaler
import os
import pywt
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

class Data_Preprocess:

    def __init__(self, data):
        self.data = data

    def process_signal(self,data, file_path):
        wavelet_coeffs = pywt.wavedec(data, 'db2')[0]  # get the array from the list (N,F,L)
        print(wavelet_coeffs.shape)
        print(wavelet_coeffs)
        wavelet_coeffs=wavelet_coeffs.reshape(wavelet_coeffs.shape[0],wavelet_coeffs.shape[2],wavelet_coeffs.shape[1])
        print(wavelet_coeffs.shape)
        #reshaped_data = np.squeeze(wavelet_coeffs, axis=1) #reshpe the data
        #unwrapped_data = np.array([[np.array(timeseries) for timeseries in sublist] for sublist in reshaped_data]) # unwrap the data from the list
        scaler = StandardScaler()  # nomarlize data
        processed_audio = np.array([[scaler.fit_transform(np.array(row).reshape(-1, 1)).flatten() for row in coeffs]
                                      for coeffs in wavelet_coeffs]) #shape (N,F,L)
        if 'D/NOK_Measurements' in file_path:
            label = [0] * processed_audio.shape[0]
        elif 'D/OK_Measurements' in file_path:
            label = [1] * processed_audio.shape[0]
        else:
            raise ValueError(f'Unexpected file path: {file_path}') #add one label for the whole dataset

        dataset = self.df_to_tensor(processed_audio,label)
        return dataset # return the TensorDataset for further data loader


    def df_to_tensor(self,features,labels):  # input dataframe return the tensor vector with the same length by using the zero padding
        features_tensor = torch.Tensor(features)
        labels_tensor = torch.Tensor(labels)
        dataset = data_utils.TensorDataset(features_tensor, labels_tensor)
        return dataset


