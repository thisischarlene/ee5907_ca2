#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part2_2b.py> { 
    context         ; 
    purpose         applying GMM to PCA; 
    used in         parent folder [A0280349Y]; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<part2_2b.py>"
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
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
from A0280349Y.part1_2c import *
from A0280349Y.part2_2a import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_1] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part2_2b
os.makedirs(dir_thisPart, exist_ok=True)


def main():
    # load the training data from <part1_1.py> 
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    
    # load the testing data from <part1_1.py> 
    X_test = np.load(os.path.join(dir_part1_1, "X_test.npy"))
    y_test = np.load(os.path.join(dir_part1_1, "y_test.npy"))
    
    #- apply GMM;
    y_train_gmm, y_test_gmm, gmm = apply_gmm(X_train, X_test, 3)

    for p in [80, 200]:
        print(f"\n--- PCA Dimension Reduction to p={p} ---")

        #- apply PCA;
        pca = PCA(n_components=p)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        #- apply GMM;
        y_train_gmm, y_test_gmm, gmm = apply_gmm(X_train_pca, X_test_pca, 3)

        #- plot 2D visualization;
        plot_2d_gmm(X_train_pca, X_test_pca, y_train_gmm, y_test_gmm, y_test, 69, f"GMM Clustering p{p}")

if __name__ == "__main__":
    main()
    
    