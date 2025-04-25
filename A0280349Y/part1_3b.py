#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part1_3b.py> { 
    context         ee5907 assignment 2; 
    purpose         apply {my_lda} and plot scatter plots; 
    used in         [A0280349Y];
    py version      3.10;
    os              windows 10; 
    ref(s)          ;       
    note            for master branch only; 
} // "<part1_3b.py>"
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
from A0280349Y.part1_1 import *
from A0280349Y.part1_3a import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_2b] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part1_3b
os.makedirs(dir_thisPart, exist_ok=True)


def plot_2d_lda(X_pca, y, my_label="", p=""):
    plt.figure(figsize=(10, 6))
    # find all the labels for the selected subjects;
    unique_labels = np.unique(y)
    
    # make sure each class is using a different colour; 
    #- get number of classes; 
    n_classes = len(unique_labels)
    #- generate the colourmap with unique colours for the classes; 
    cmap = cm.get_cmap('nipy_spectral', n_classes)
    norm = mcolors.Normalize(vmin=0, vmax=n_classes-1)
    for i, label in enumerate(unique_labels):
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], color=cmap(norm(i)), label=f"Class {label}", alpha=0.6, s=20)
        
    idx_mine = y == my_label
    plt.scatter(X_pca[idx_mine, 0], X_pca[idx_mine, 1], color='black', marker='x', s=100, label=f"Subject {my_label}")
    plt.title(f"PCA Projection (2D, p = {p}) with Subject {my_label} Highlighted")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc='center left', fontsize='small', markerscale=1.5, bbox_to_anchor=(1.02, 0.5))
    plt.grid(True)
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt_name = f"PCA_2D_Subject{my_label}_p{X_pca.shape[1]}.png"
    plt.savefig(os.path.join(dir_thisPart, plt_name), dpi=300, bbox_inches="tight")
    print(f"2D PCA Scatter Plot {plt_name} saved in {dir_thisPart} ... ")
    #plt.show()
        

def plot_3d_lda(X_pca, y, my_label="", p=""): 
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')
    # find all the labels for the selected subjects;
    unique_labels = np.unique(y)
    
    # make sure each class is using a different colour; 
    #- get number of classes; 
    n_classes = len(unique_labels)
    #- generate the colourmap with unique colours for the classes; 
    cmap = cm.get_cmap('nipy_spectral', n_classes)
    norm = mcolors.Normalize(vmin=0, vmax=n_classes-1)
    for i, label in enumerate(unique_labels):
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], color=cmap(norm(i)), label=f"Class {label}", alpha=0.6, s=20)
        
    idx_mine = y == my_label
    plt.scatter(X_pca[idx_mine, 0], X_pca[idx_mine, 1], color='black', marker='x', s=100, label=f"Subject {my_label}")
    ax.set_title(f"PCA Projection (3D, p = {p}) with Subject {my_label} Highlighted")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    #-rescale so that it doesnt look congested; 
    ax.view_init(elev=40, azim=45)
    ax.auto_scale_xyz(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])   
    ax.set_xlim(-2000, 3600)
    ax.set_ylim(-2300, 2300)
    ax.set_zlim(-1500, 1300)
    """
    #- to find the actual limits of the plot in 3d;
    print(f"xlimits (PC1): {X_pca[:, 0].min():.2f}, {X_pca[:, 0].max():.2f}")
    print(f"ylimits (PC2): {X_pca[:, 1].min():.2f}, {X_pca[:, 1].max():.2f}")
    print(f"zlimits (PC3): {X_pca[:, 2].min():.2f}, {X_pca[:, 2].max():.2f}")
    """

    plt.legend(loc='center left', fontsize='small', markerscale=1.5, bbox_to_anchor=(1.02, 0.5))
    plt.grid(True)
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt_name = f"PCA_3D_Subject{my_label}_p{X_pca.shape[1]}.png"
    plt.savefig(os.path.join(dir_thisPart, plt_name), dpi=300, bbox_inches="tight")
    print(f"3D PCA Scatter Plot {plt_name} saved in {dir_thisPart} ... ")
    #plt.show()

def main():
    # load the data from <part1_1.py> 
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    
    # plot the eigenfaces; 
    #- when p = 2, 
    plot_eigenfaces(X_train, 2)
    #- when p = 3, 
    plot_eigenfaces(X_train, 3)
    
    
    # plot the 2d pca; 
    #- apply PCA with p=2; 
    _, X_pca_2d = my_pca(X_train, p=2)
    #- plot the pca results, highlighting the mock_subject; 
    plot_2d_pca(X_pca_2d, y_train, 69, 2)
    
    
    # plot the 3d pca; 
    #- apply PCA with p=3; 
    _, X_pca_3d = my_pca(X_train, p=3)
    plot_3d_pca(X_pca_3d, y_train, 69, 3)

if __name__=="__main__":
    main()
 