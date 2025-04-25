#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part2_1a.py> { 
    context         ; 
    purpose         applying KNN to PCA; 
    used in         parent folder [A0280349Y]; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<part2_1a.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""


import os
import datetime
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mping
import random
from A0280349Y.config import *
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from A0280349Y.part1_2c import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_1] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part2_1a
os.makedirs(dir_thisPart, exist_ok=True)

def apply_knn(X_train, y_train, X_test, y_test, k=1):
    #- initialise KNN classifier; 
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #- train the classifier; 
    knn.fit(X_train, y_train)
    
    #- predict the labels based on test data;
    y_pred = knn.predict(X_test)
    
    #- compute the accuracy; 
    accuracy_knn = accuracy_score(y_test, y_pred)
    print(f"accuracy for k={k}: {accuracy_knn: 0.4f}")
    
    return accuracy_knn

def main():
    # load the training data from <part1_1.py> 
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    
    
    # load the PCA-transformed data for p=80, p=200;
    X_pca_80 = np.load(os.path.join(dir_part1_2c, "X_pca_80.npy"))
    eigvecs_80 = np.load(os.path.join(dir_part1_2c, "eigvecs_80.npy"))
    X_pca_200 = np.load(os.path.join(dir_part1_2c, "X_pca_200.npy"))
    eigvecs_200 = np.load(os.path.join(dir_part1_2c, "eigvecs_200.npy"))

    
    # load the testing data from <part1_1.py> 
    X_test = np.load(os.path.join(dir_part1_1, "X_test.npy"))
    y_test = np.load(os.path.join(dir_part1_1, "y_test.npy"))
    
    # to get PCA-transformed test data;
    X_pca_80_test = X_test @ eigvecs_80
    X_pca_200_test = X_test @ eigvecs_200  
    
    # apply KNN for PCA-transformed test data;
    apply_knn(X_pca_80, y_train, X_pca_80_test, y_test, k=1)
    apply_knn(X_pca_200, y_train, X_pca_200_test, y_test, k=1)

if __name__ == "__main__":
    main()
    
    