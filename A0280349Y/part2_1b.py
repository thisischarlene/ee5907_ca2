#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part2_1b.py> { 
    context         ; 
    purpose         applying KNN to LDA; 
    used in         parent folder [A0280349Y]; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<part2_1b.py>"
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
from A0280349Y.part1_3c import *
from A0280349Y.part2_1a import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_1] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part2_1b
os.makedirs(dir_thisPart, exist_ok=True)


def main():
    # load the training data from <part1_1.py> 
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    
    
    # load the LDA-transformed data for p=9, p=15;
    X_lda_9 = np.load(os.path.join(dir_part1_3c, "X_lda_9.npy"))
    eigvecs_9 = np.load(os.path.join(dir_part1_3c, "eigvecs_9.npy"))
    X_lda_15 = np.load(os.path.join(dir_part1_3c, "X_lda_15.npy"))
    eigvecs_15 = np.load(os.path.join(dir_part1_3c, "eigvecs_15.npy"))
    
    # load the testing data from <part1_1.py> 
    X_test = np.load(os.path.join(dir_part1_1, "X_test.npy"))
    y_test = np.load(os.path.join(dir_part1_1, "y_test.npy"))
    
    # to get PCA-transformed test data;
    X_lda_9_test = X_test @ eigvecs_9
    X_lda_15_test = X_test @ eigvecs_15  
    
    # apply KNN for PCA-transformed test data;
    apply_knn(X_lda_9, y_train, X_lda_9_test, y_test, k=1)
    apply_knn(X_lda_15, y_train, X_lda_15_test, y_test, k=1)

if __name__ == "__main__":
    main()
    
    