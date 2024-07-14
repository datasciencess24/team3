# Data Science 2024



## Overview

- This is our Gaussian Mixture Model (GMM) for grinding anomaly detection.<br>
- It boasts a perfect F2-score on clean and noisy samples between 70 and 2000 in size, hence being provably useful for smaller datasets.
- By extracting only a select few features from the raw time-series data, it both achieves speed and allows for observed anomalies to be more easily linked to their root causes in real time, striking the ideal balance between accuracy, interpretability and efficiency.<br>
- Our experiments demonstrate that this GMM performs significantly better than other explainable ML alternatives under these low-data conditions, namely an isolation forest, a one-class support vector machine and an autoencoder.


## Structure

- /frontend: contains the necessary files of the dashboard. The data is read from frontend/data/data. a local python server must be running via cors_http_server.py to access the data. Open Dashboard.html for the UI
- /tuple_loader.py: loads the data files grinding_ok_train.json and grinding_test.json
- /gmm_for_tuples.py: code for the Gaussian Mixture Model (GMM)
- /run_gmm.py: runs the GMM
- /ae_for_tuples.py, /if_for_tuples.py, /oc_svm_for_tuples.py: code for the comparison models, i.e. the autoencoder, Isolation Forest and one-class support vector machine respectively
- /run_ae.py, /run_if.py, /run_oc_svm.py: code to run the three comparison models 

## Data

This is stored in the same directory as the code, in two separate JSON files:
- /grinding_ok_train.json: contains the flattened time-series data used for training; comprises 2000 recordings of normal grinding
- /grinding_test.json: contains the flattened time-series data used for testing; comprises 29 recordings of normal grinding and 29 of simulated anomalies in grinding

## Project status

This project is a work in progress.
