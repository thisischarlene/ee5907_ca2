#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part1_2c.py> { 
    context         ee5907 assignment 2; 
    purpose         apply {my_PCA} for PCs=80; 
    used in         [A0280349Y];
    py version      3.10;
    os              windows 10; 
    ref(s)          ;       
    note            for master branch only; 
} // "<part1_2c.py>"
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
from A0280349Y.part1_2a import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_2b] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part1_2c
os.makedirs(dir_thisPart, exist_ok=True)


def main():
    # load the data from <part1_1.py> 
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    
    # apply {my_pca} with p=80; 
    eigvecs_80, X_pca_80 = my_pca(X_train, p=80)
    #- save as binary file; 
    np.save(os.path.join(dir_thisPart, "X_pca_80.npy"), X_pca_80)
    np.save(os.path.join(dir_thisPart, "eigvecs_80.npy"), eigvecs_80)
   

if __name__=="__main__":
    main()
 