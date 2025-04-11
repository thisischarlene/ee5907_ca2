#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<config.py> { 
    context         config file for assignment; 
    purpose         custom functions used throughout the assignment; 
    used in         all files except <__init__.py>; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<config.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""

import os 
import datetime
import csv
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# get absolute path of <config.py>; 
dir_base = os.path.dirname(os.path.abspath(__file__))

# declare main output folder;
#- generate timestamp; 
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#- create new directory for results; 
dir_results = os.path.join(dir_base, "results", timestamp)
# dir_results = os.path.join(dir_base, "results")
os.makedirs(dir_results, exist_ok=True)

# from main output folder, create subfolders;
#-- folder names are based on assignment parts; 
dir_part1a = os.path.join(dir_results, "part1a")
dir_part1b = os.path.join(dir_results, "part1b")
dir_part1c = os.path.join(dir_results, "part1c")
dir_part2a = os.path.join(dir_results, "part2a")
dir_part2b = os.path.join(dir_results, "part2b")
#--- part2c = comparison of 2a and 2b; 
dir_part2c = os.path.join(dir_results, "part2c")


# set random seed value; 
seed = 49


# to log performance metrics; 
def log_performance(assignment_part, accuracy, precision, recall, filename=f"{dir_results}/performanceMetrics_{timestamp}.csv"):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="") as file: 
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(["Assignment Part", "Accuracy", "Precision", "Recall"]) 
            
        # append new row after each call; 
        writer.writerow([assignment_part, accuracy, precision, recall])
        
        screen_filename = f"performanceMetrics_{timestamp}.csv"
        
        print_with_plus(f"performance metrics for {assignment_part} logged into {screen_filename}...")


# styles for print to screen; 
def print_with_star(message):
    border = '*' * (len(message) + 4)
    print(border)
    print(f"* {message} *")
    print(border)
    
def print_with_hash(message):
    border = '#' * (len(message) + 4)
    print(border)
    print(f"# {message} #")
    print(border)
    
def print_with_plus(message):
    border = '+' * (len(message) + 4)
    print(border)
    print(f"+ {message} +")
    print(border)


# default plot settings; 
def default_pltSettings():
    plt.figure(figsize=(8,6), dpi=600)
    plt.grid(True, linestyle="-", alpha=0.5, color="gray")
    
    
def save_plt(name_figure):
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(name_figure, bbox_inches="tight")
    

# plot the ROC curve;
def plot_roc(y_true, y_pred, line_colour, plot_title, name_figure):
    fpr, tpr, _ = roc_curve(y_true, y_pred.flatten())
    roc_auc = auc(fpr, tpr) 
    
    default_pltSettings()
    plt.plot(fpr, tpr, color=line_colour, label=f"ROC Curve (AUC = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color=line_colour)
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(plot_title)
    
    save_plt(name_figure)
    
# pre-defined colours;
colour1 = (118/255, 105/255, 89/255)
colour2 = (93/255, 95/255, 125/255)
colour3 = (34/255, 107/255, 143/255)
colour4 = (81/255, 31/255, 60/255)
colour5 = (34/255, 19/255, 34/255)