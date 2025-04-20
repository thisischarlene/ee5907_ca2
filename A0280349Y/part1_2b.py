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
from A0280349Y import *


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
        plt.title(f"Visualising Eigenface {i+1}")
        plt.axis('off')
        plt.colorbar()
        plt.show()



if __name__=="__main__":
    """
    eigvecs_top, _ = my_pca(images_train, 3)
    visualise_eignfaces(eigvecs_top, num_faces=3, title_prefix="PCA")
    """
    
    print(images_train.shape)