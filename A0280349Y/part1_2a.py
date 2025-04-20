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


def my_pca(X, p): 
    # center the data; 
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean 
    
    # compute co-varaiance matrix;
    matrix_cov = np.cov(X_centered, rowvar=False)
    
    # compute eigen decomposition based on co-variance matrix;
    eigvals, eigvecs = np.linalg.eigh(matrix_cov)
    
    # sort eigenvectors by descending eignvalues to keep top p;  
    idx_sorted = np.argsort(eigvals)[::-1] 
    eigvecs_sorted = eigvecs[:, idx_sorted]
    
    # select the top p eigenvectors; 
    W_p = eigvecs_sorted[:, p]
    
    # project the data; 
    X_pca = X_centered @ W_p
    return W_p, X_pca 

    

# if __name__=="__main__":