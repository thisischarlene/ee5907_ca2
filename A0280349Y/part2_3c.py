#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part2_3b.py> { 
    context         ; 
    purpose         applying SVM for LDA; 
    used in         parent folder [A0280349Y]; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<part2_3b.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""


import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from A0280349Y.config import *
from A0280349Y.part2_3a import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_1] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part2_3b
os.makedirs(dir_thisPart, exist_ok=True)


def reduce_lda(X_train, y_train, X_test, p):
    lda = LDA(n_components=p)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda


def main():
    X_train, y_train, X_test, y_test = load_data()
    C_values = [1e-2, 1e-1, 1]
    for p in [80, 200]:
        X_train_pca, X_test_pca = reduce_lda(X_train, X_test, p)
        run_experiments(X_train_pca, y_train, X_test_pca, y_test, p, C_values)

if __name__ == "__main__":
    main()
    
    