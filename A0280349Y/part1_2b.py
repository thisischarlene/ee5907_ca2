#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part1_2b.py> { 
    context         ee5907 assignment 2; 
    purpose         apply {my_PCA} and plot scatter plots; 
    used in         [A0280349Y];
    py version      3.10;
    os              windows 10; 
    ref(s)          ;       
    note            for master branch only; 
} // "<part1_2b.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""

import os
import datetime
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from A0280349Y.config import *
from A0280349Y.part1_1 import *
from A0280349Y.part1_2a import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_2b] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part1_2b
os.makedirs(dir_thisPart, exist_ok=True)

def visualise_eignfaces(eigvecs, num_faces, title_prefix=""):
    for i in range(num_faces):
        # size of image is 32x32;
        eigface = eigvecs[:, 1].reshape(32, 32)
        plt.imshow(eigface, cmap="gray")
        plt.title(f"{title_prefix} Visualising Eigenface {i+1}")
        plt.axis('off')
        plt.colorbar()
        plt.show()

def main():
    # load the data from <part1_1.py> 
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    
    # when p = 2, 
    p=2
    eigvecs_2, _ = my_pca(X_train, p=p)
    visualise_eignfaces(eigvecs_2, num_faces=p, title_prefix=f"PCA (p={p})")
    
    
    # when p = 3, 
    p=3
    eigvecs_3, _ = my_pca(X_train, p=p)
    visualise_eignfaces(eigvecs_3, num_faces=p, title_prefix=f"PCA (p={p})")

if __name__=="__main__":
    main()
 