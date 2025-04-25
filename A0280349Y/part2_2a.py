#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part2_2a.py> { 
    context         ; 
    purpose         applying GMM to PCA; 
    used in         parent folder [A0280349Y]; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<part2_2a.py>"
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
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from A0280349Y.part1_1 import *
from A0280349Y.part1_2c import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_1] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part2_2a
os.makedirs(dir_thisPart, exist_ok=True)

def apply_gmm(X_train, X_test, n_components=3):
    #- initialise KNN classifier; 
    gmm = GaussianMixture(n_components=3, random_state=seed)
    
    #- train the classifier; 
    gmm.fit(X_test)
    
    #- predict the labels based on test data;
    y_pred = gmm.predict(X_test)
    
    return y_pred, gmm 

def plot_2d_gmm(X_train, X_test, y_pred, my_label, title="GMM  Clustering" ):
    #- reduce data to 2D using PCA for visualisation;
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)
    
    #- plot the 2d -- copy from <part1_2b.py>; 
    plt.figure(figsize=(10, 6))
    # find all the labels for the selected subjects;
    unique_labels = np.unique(y)
    
    #- make sure each class is using a different colour; 
    #-- get number of classes; 
    n_classes = len(unique_labels)
    #-- generate the colourmap with unique colours for the classes; 
    cmap = cm.get_cmap('nipy_spectral', n_classes)
    norm = mcolors.Normalize(vmin=0, vmax=n_classes-1)
    
    #-- plot training data;  
    for i, label in enumerate(unique_labels): 
        idx = y_pred == label
        plt.scatter(X_train_2d[idx, 0], X_train_2d[idx, 1], color=cmap(norm(i)), label=f"Train Class {label}", alpha=0.6, s=20)
        
    #-- plot testing data;  
    for i, label in enumerate(unique_labels): 
        idx = y_pred == label
        plt.scatter(X_test_2d[idx, 0], X_test_2d[idx, 1], color=cmap(norm(i)), label=f"Test Class {label}", alpha=0.6, s=20)
    
    #-- highlight {mock_subject};    
    idx_mine = y_pred == my_label
    plt.scatter(X_test_2d[idx_mine, 0], X_test_2d[idx_mine, 1], color='black', marker='x', s=100, label=f"Subject {my_label}")
    
    plt.title(f"{title} (2D) with Subject {my_label} Highlighted")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc='center left', fontsize='small', markerscale=1.5, bbox_to_anchor=(1.02, 0.5))
    plt.grid(True)
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt_name = f"{title}_2D_Subject{my_label}.png"
    plt.savefig(os.path.join(dir_thisPart, plt_name), dpi=300, bbox_inches="tight")
    print(f"2D GMM Clustering Scatter Plot {plt_name} saved in {dir_thisPart} ... ")
    #plt.show()
    plt.close()
    
     
def main():
    #- load the training data from <part1_1.py> 
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    
    
    #- load the PCA-transformed data for p=80, p=200;
    X_pca_80 = np.load(os.path.join(dir_part1_2c, "X_pca_80.npy"))
    eigvecs_80 = np.load(os.path.join(dir_part1_2c, "eigvecs_80.npy"))
    X_pca_200 = np.load(os.path.join(dir_part1_2c, "X_pca_200.npy"))
    eigvecs_200 = np.load(os.path.join(dir_part1_2c, "eigvecs_200.npy"))

    
    #- load the testing data from <part1_1.py> 
    X_test = np.load(os.path.join(dir_part1_1, "X_test.npy"))
    y_test = np.load(os.path.join(dir_part1_1, "y_test.npy"))
    
    #- to get PCA-transformed test data;
    X_pca_80_test = X_test @ eigvecs_80
    X_pca_200_test = X_test @ eigvecs_200  
    
    #- apply KNN for PCA-transformed test data;
    apply_gmm(X_pca_80, y_train, X_pca_80_test, y_test, k=1)
    apply_gmm(X_pca_200, y_train, X_pca_200_test, y_test, k=1)

if __name__ == "__main__":
    main()
    
    