# Data Science 2024



## Overview

- This is our explainable fusion model for grinding anomaly detection.<br>
- It allows for observed anomalies to be more easily linked to their root causes in real time, striking the ideal balance between accuracy, interpretability and efficiency.<br>
- It comprises a convolutional neural network (CNN) and decision tree (DT).<br>
- Our experiments demonstrate that this fusion model performs significantly better than other explainable alternatives, namely a parallel CNN-DT fusion model, CNN with heat map, DT used alone and regularised logistic regression.


## Structure

**/Data Loader**  includes methods for loading and preprocessing data.
- /Config.py: stores all the parameters, such as the model hyperparameters, training parameters and configuration parameters
- /Data_Module.py: handles loading data for use in a PyTorch model. It includes functionalities for reading data from multiple files, concatenating them into a single DataFrame, and preparing them for training with a custom collate function
- /Dataset.py: customized structure of the time series dataset
- /data_preprocessing.py: handles the preprocessing of time series data using wavelet transformation and standardization. It includes functionalities for reading data, applying wavelet decomposition, standardizing the data, and converting it to a PyTorch Tensordataset


**/Models** contains both of our models that will be trained and tested both jointly and separately. As this project is not yet complete, certain parameters are still to be determined, such as the dimensionality of output for the CNN and the depth of the DT.

**/Trainer** includes all methods that are used to train the models.
- /trainCNN.py: trains a Convolutional Neural Network (CNN) model on time series data using PyTorch; includes functionalities for training the model and saving the trained model

**/Data** contains two folders, one containing normal ('OK') data and the other anomalous ('NOK') data.

**main.py**: output will be a trained CNN model (if there is no pretrained CNN) stored with the name 'Timeseries_Conv_model_1.pkl' and the results of the DecisionTree classifier including F1-score, precision and recall


## Data

The training data consists of 29 normal ('OK') and 29 anomalous ('NOK') datapoints, given in the data folder.
Here we only uploaded part of the data as samples.

## Project status

This project is a work in progress.
