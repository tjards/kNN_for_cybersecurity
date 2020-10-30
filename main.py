#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calls k-NN to classifiy malicious network traffic

See the README for more details

@author: tjards
P. Travis Jardine, PhD
Adjunct assistant professor
Department of Electrical and Computer Engineering
Royal Military College of Canada
peter.jardine@rmc.ca

dated: 30 Oct 2020

"""


#%% Import useful stuff 
# ------------------------------------------------------
import server       # this server runs the learning code
import numpy as np  # import numpy library


#%% COLLECT data
# Here we simulate data collection by pointing to a few files
# See README for more info on the dataset
# -----------------------------------------------------------
data_train = 'data/train_inputs.csv'          # training set data
labels_train = 'data/train_outputs.csv'       # traing set labels
data_test = 'data/test_inputs.csv'            # test set data
labels_test = 'data/test_outputs.csv'         # test set labels


#%% SELECT features
# Here we decide which features from the data set to use 
# There are many ways to do this
# I have selected these features using a special algo called FOS
# I saved the index for the features in a special file that we will load
# -----------------------------------------------------------------------
selected_features = np.load('selected_features.npy')


#%% RUN the k-NN algo
# I have developed a short script that runs k-NN on our pretend server
# it needs the indicies for the features and the data paths defined above
# -----------------------------------------------------------------------
classif_accuracy_knn, time_knn_train, time_knn_classify = server.run_knn(selected_features, data_train, labels_train, data_test, labels_test)


#%% PRESENT the results
# ----------------------
print('====== RESULTS ========')
print('k-NN classified with ', classif_accuracy_knn*100, '%% accuracy')
print('The training took ',time_knn_train,'secs')
print('The classification took ',time_knn_classify,'secs')
