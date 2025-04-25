#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part1_3a.py> { 
    context         ee5907 assignment 2; 
    purpose         write custom LDA function; 
    used in         [A0280349Y];
    py version      3.10;
    os              windows 10; 
    ref(s)          ;       
    note            for master branch only; 
} // "<part1_3a.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""

import os
import datetime
import numpy as np 
import matplotlib.pyplot as plt
import random
from A0280349Y.config import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_2a] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part1_3a
os.makedirs(dir_thisPart, exist_ok=True)

def my_lda(X, y, p):
    #-find all unique class labels aka {subjects_labels}; 
    classes = np.unique(y)
    #- get number of features; 
    n_features = X.shape[1]
    #- to find the distance between each class' local mean and the global mean; 
    mean_overall = np.mean(X, axis=0)
    
    #- compute the noise variance within the classes;
    scatter_within = np.zeros=((n_features, n_features))
    #- compute the signal variance between the classes;
    scatter_between = np.zeros=((n_features, n_features))
    
    #- processing one class at a time; 
    for cls in classes:
        #-- get all images from this class aka the subject; 
        X_classes = X[y == cls]
        #-- mean of all faces; 
        mean_classes = np.mean(X_classes, axis=0)
        #-- compute within-class scatter; how much it spreads around local class mean; 
        scatter_within += np.dot((X_classes - mean_classes).T, (X_classes - mean_classes))
        #-- calculate how many samples in the class;
        n_classes = X_classes.shape[0]
        #-- compute the difference between local mean and global mean;
        mean_diff = (mean_classes - mean_overall).reshape(-1, 1)
        #-- compute between-class scatter; distance between local class mean and global mean via class size weights;
        scatter_between += n_classes * np.dot(mean_diff, mean_diff.T)

def main():
    print("LDA function, {my_lda}, has been defined ...")

if __name__=="__main__":
    main()