#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part2_3a.py> { 
    context         ; 
    purpose         applying SVM for classification; 
    used in         parent folder [A0280349Y]; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<part2_3a.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""


import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from A0280349Y.config import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_1] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part2_3a
os.makedirs(dir_thisPart, exist_ok=True)

def load_data():
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    X_test = np.load(os.path.join(dir_part1_1, "X_test.npy"))
    y_test = np.load(os.path.join(dir_part1_1, "y_test.npy"))
    return X_train, y_train, X_test, y_test


def train_svm(X_train, y_train, C):
    clf = LinearSVC(C=C, max_iter=30000, random_state=seed)
    clf.fit(X_train, y_train)
    return clf
    
    
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc


def run_experiments(X_train, y_train, X_test, y_test, C_values):
    results = []
    print("\n====== Linear SVM Classification ======")
    for C in C_values:
        print(f"\n--- C = {C} ---")
        clf = train_svm(X_train, y_train, C)
        train_acc = evaluate_model(clf, X_train, y_train)
        test_acc = evaluate_model(clf, X_test, y_test)

        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Testing Accuracy : {test_acc:.4f}")

        results.append((C, train_acc, test_acc))
    return results

def main():
    X_train, y_train, X_test, y_test = load_data()
    C_values = [1e-2, 1e-1, 1]
    run_experiments(X_train, y_train, X_test, y_test, C_values)

if __name__ == "__main__":
    main()
    
    