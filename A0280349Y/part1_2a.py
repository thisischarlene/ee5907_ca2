#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part1_2a.py> { 
    context         ee5907 assignment 2; 
    purpose         write custom PCA function; 
    used in         [A0280349Y];
    py version      3.10;
    os              windows 10; 
    ref(s)          ;       
    note            for master branch only; 
} // "<part1_2a.py>"
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
dir_thisPart = dir_part1_2a
os.makedirs(dir_thisPart, exist_ok=True)

def my_pca(X, p): 
    # centre the data by subtracting the mean; 
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean 
    
    # compute co-variance matrix;
    matrix_cov = np.cov(X_centered, rowvar=False)
    
    # compute eigen decomposition based on co-variance matrix;
    eigvals, eigvecs = np.linalg.eigh(matrix_cov)
    
    # sort eigenvectors by descending eignvalues to keep top p;  
    idx_sorted = np.argsort(eigvals)[::-1] 
    eigvecs_top = eigvecs[:, idx_sorted[:p]]
    
    
    # dot multiply the X_centered with the top p to get PCA-transformed data; 
    X_pca = np.dot(X_centered, eigvecs_top)
    return  eigvecs_top, X_pca 


def main():
    print("PCA function, {my_pca}, has been defined ...")

if __name__=="__main__":
    main()