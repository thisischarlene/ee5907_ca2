#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part1_3c.py> { 
    context         ee5907 assignment 2; 
    purpose         apply {my_lda} for p=9 + p =15; 
    used in         [A0280349Y];
    py version      3.10;
    os              windows 10; 
    ref(s)          ;       
    note            for master branch only; 
} // "<part1_3c.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""

import os
import datetime
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
import random
from A0280349Y.config import *
from A0280349Y.part1_3a import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_2b] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part1_3c
os.makedirs(dir_thisPart, exist_ok=True)


def main():
    # load the data from <part1_1.py> 
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    
    # apply {my_lda} with p=9; 
    eigvecs_9, X_lda_9 = my_lda(X_train, y_train, p=9)
    #- save as binary file; 
    np.save(os.path.join(dir_thisPart, "X_lda_9.npy"), X_lda_9)
    np.save(os.path.join(dir_thisPart, "eigvecs_9.npy"), eigvecs_9)
    
    # apply {my_pca} with p=15; 
    eigvecs_15, X_lda_15 = my_lda(X_train, y_train, p=15)
    #- save as binary file; 
    np.save(os.path.join(dir_thisPart, "X_lda_15.npy"), X_lda_15)
    np.save(os.path.join(dir_thisPart, "eigvecs_15.npy"), eigvecs_15)
   

if __name__=="__main__":
    main()
 