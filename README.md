# Data Science 2024



## Overview

- Our explainable fusion model for grinding anomaly detection allows for observed anomalies to be more easily linked to their root causes in real time, striking the ideal balance between accuracy, interpretability and efficiency.  
- It comprises a convolutional neural network (CNN) and decision tree (DT).
- Our experiments demonstrate that this fusion model performs significantly better than other explainable alternatives, namely a parallel CNN-DT fusion model, CNN with heat map, DT used alone and regularised logistic regression.


## Structure

**/Data Loader**  includes  methods for loading data and preprocessing them.
- /Config.py: stores all the parameters, such as the model hyperparameters, training parameters and configuration parameters
- /Data_Module.py: handles loading data for use in a PyTorch model. It includes functionalities for reading data from multiple files, concatenating them into a single DataFrame, and preparing them for training with a custom collate function
- /Dataset.py: customized structure of the time series dataset
- /data_preprocessing.py: handles the preprocessing of time series data using wavelet transformation and standardization. It includes functionalities for reading data, applying wavelet decomposition, standardizing the data, and converting it to a PyTorch Tensordataset


**/Models** contains both of our models that will be trained and tested both jointly and separately.

**/Trainer** includes all methods that are used to train the models.
- /trainCNN.py: trains a Convolutional Neural Network (CNN) model on time series data using PyTorch; includes functionalities for training the model and saving the trained model

**/Data** contains both the data-files that are classified as ok and those that are classified as not-ok.

**main.py**: output will be a trained CNN model (if there is no pretrained CNN) stored with the name 'Timeseries_Conv_model_1.pkl' and the results of the DecisionTree classifier including F1-score, precision and recall

## Data

The training date consits of 29 normal/ok data-sets and 29 not-ok data-sets, given in the data folder.

## Project status
This project still is not finished and classified as work in progress.
