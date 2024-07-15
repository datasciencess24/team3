# Data Science 2024



## Overview

- This is our convolutional autoencoder-Gaussian mixture model (CAE-GMM) fusion model  for grinding anomaly detection.
- It boasts a perfect F2-score on clean and noisy samples between 70 and 2000 in size, hence being provably useful for smaller datasets.
- By extracting only a select few features from the raw time-series data, it both achieves speed and allows for observed anomalies to be more easily linked to their root causes in real time, striking the ideal balance between accuracy, interpretability and efficiency.
- Our experiments demonstrate that this CAE-GMM performs significantly better than other explainable ML alternatives under these low-data conditions, namely an isolation forest, a one-class support vector machine and an autoencoder.


## Structure
- /cae_gmm_140724: 
-- /DataLoader.py:  Loads data from files
-- /data_preprocessing.py: Handles the preprocessing of the time series data through sampling, standardization and segmentation
-- /Model.py: Contain the model Convolutional Autoencoder and Gaussian Mixture Model(GMM)
-- /train.py: train the model on time series data using PyTorch. It includes functionalities for training the model, and saving the trained model.
-- /main.py: output will be a trained CAE-GMM model(if there is no pretrained model),the proportions of anomalous data detected in the training set, two separate JSON files containing normal and anomalous results after the anomaly detection.
For both training and test data, the code a directory structure where training files are located within subdirectories named ‘raw’ (or containing ‘raw’ in their path), potentially among other directories. It specifically looks for two types of parquet files within these raw directories:
- 1. A file named ‘Sampling2000KHz_AEKi-0.parquet’ for acoustic emissions (AE) data.
- 2. A file named ‘Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet’ for electrical current (EC) data.

To accommodate any deviations, the necessary changes will need to be made to the load_training_files and load_test_files methods DataLoader.py module.



- /frontend: contains the necessary files of the dashboard. The data is read from frontend/data/data. a local python server must be running via cors_http_server.py to access the data. Open Dashboard.html for the UI
- /tuple_loader.py: loads the data files grinding_ok_train.json and grinding_test.json
- /gmm_for_tuples.py: code for the Gaussian Mixture Model (GMM)
- /run_gmm.py: runs the GMM
- /ae_for_tuples.py, /if_for_tuples.py, /oc_svm_for_tuples.py: code for the comparison models, i.e. the autoencoder, Isolation Forest and one-class support vector machine respectively
- /run_ae.py, /run_if.py, /run_oc_svm.py: code to run the three comparison models 

## Data

This is stored in /new model data in two separate JSON files:
- /grinding_ok_train.json: contains the flattened time-series data used for training; comprises 2000 recordings of normal grinding
- /grinding_test.json: contains the flattened time-series data used for testing; comprises 29 recordings of normal grinding and 29 of simulated anomalies in grinding

## Project status

This project is finished. Adjustments can be made in the future.
