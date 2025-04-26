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

def map_gmm_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for i in np.unique(y_pred):
        mask = (y_pred == i)
        labels[mask] = mode(y_true[mask], keepdims=True)[0]
    return labels

def evaluate_gmm(y_true, y_pred, title=""):
    y_pred_mapped = map_gmm_labels(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred_mapped)
    acc = np.trace(cm) / np.sum(cm)

    #-- save confusion matrix;
    plt.figure(figsize=(10, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {title}")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt_name = f"ConfusionMatrix_{title}.png"
    plt.savefig(os.path.join(dir_thisPart, plt_name), dpi=300, bbox_inches="tight")
    print(f"Confusion Matrix {plt_name} saved in {dir_thisPart} ...")
    #plt.show()
    plt.close()

    print(f"Accuracy: {acc:.4f}")
    return cm, acc
    


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
        plot_2d_gmm(X_train_pca, X_test_pca, y_train_gmm, y_test_gmm, 69, f"GMM Clustering p{p}")

        #- evaluate;
        evaluate_gmm(y_test, y_test_gmm, title=f"GMM_p{p}")

if __name__ == "__main__":
    main()
    
    