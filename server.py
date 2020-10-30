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

#%% IMPORT stuff 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time


#%% This function runs k-Nearest Neighbours for binary classification 
# It has the option to normalize the datasets, which is usually a good idea
# -------------------------------------------------------------------------
def run_knn(basis_sel_ind, path_basis_can, path_outputs_train, path_inputs_test, path_outputs_test):
    
    normalize = 1   # (yes = 1, no = 0 ) Normalizing speeds things up
    print('Server has access to all data and will run K-NN')
    print('This should take about 2 minutes ...')
    
    # prep the training set
    basis_can=pd.DataFrame(pd.read_csv(path_basis_can,header=0,index_col=0))
    outputs_train=pd.DataFrame(pd.read_csv(path_outputs_train,header=None,names=['Label']))
    if normalize == 1:
        scaler = StandardScaler()
        StandardScaler(copy=True, with_mean=True, with_std=True)
        basis_can=np.array(basis_can, dtype='float64')
        basis_can=scaler.fit_transform(basis_can)
    
    # prep the test set
    inputs_test=pd.DataFrame(pd.read_csv(path_inputs_test,header=0,index_col=0))
    outputs_test=pd.DataFrame(pd.read_csv(path_outputs_test,header=None,names=['Label']))
    if normalize ==1:
        scaler = StandardScaler()
        StandardScaler(copy=True, with_mean=True, with_std=True)
        inputs_test=np.array(inputs_test, dtype='float64')
        inputs_test=scaler.fit_transform(inputs_test)
    
    # define the classifier 
    neigh = KNeighborsClassifier(n_neighbors=3)
     
    # train on the (labelled) training set 
    basis_knn=basis_can[:,basis_sel_ind] # pull out just the features we want 
    basis_knn=np.concatenate((np.ones((outputs_train.shape[0],1)) ,basis_knn),axis=1)
    y_knn=outputs_train
    y_knn=np.ravel(np.array(outputs_train, dtype='int'))
    print('training ...')
    tic1=time.process_time()        # start clock
    neigh.fit(basis_knn, y_knn)     # train
    toc1=time.process_time()        # stop clock
    time_knn_train=toc1-tic1        # compute time
    
    # classifiy the test set
    basis_knn_test=inputs_test[:,basis_sel_ind] # pull out just the features we want
    basis_knn_test=np.concatenate((np.ones((outputs_test.shape[0],1)) ,basis_knn_test),axis=1)
    print('classifying ... ')
    tic2=time.process_time()                        # start clock
    predicted_knn=neigh.predict(basis_knn_test)     # classify the data (predict)
    toc2=time.process_time()                        # stop clock
    time_knn_predict=toc2-tic2                      # compute time
    
    # compare the k-NN predicted classifications against the labels 
    classif_error_knn=np.array(predicted_knn,ndmin=2).transpose()-outputs_test
    classif_accuracy_knn=1-np.count_nonzero(classif_error_knn)/len(classif_error_knn)
    #print('classification accuracy =',classif_accuracy_knn*100,'%%')
    
    return classif_accuracy_knn, time_knn_train, time_knn_predict
