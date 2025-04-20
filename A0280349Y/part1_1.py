#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part1_1.py> { 
    context         ee5907 assignment 2; 
    purpose         dataset preparation; 
    used in         [A0280349Y];
    py version      3.10;
    os              windows 10; 
    ref(s)          ;       
    note            for master branch only; 
} // "<part1_1.py>"
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


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_1] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part1_1
os.makedirs(dir_thisPart, exist_ok=True)

# select 25 out of the 68; 
def select_main(seed, subjects_total=68, sample_size=25):
    # set the seed based on <config.py>; 
    #- using random from python instead of numpy -- based on requirements;
    random.seed(seed)
    return random.sample(range(1, subjects_total +1), sample_size)


# select 1 out of the (68-25); 
def select_mock(subjects_total, subjects_selected):
    subjects_remaining = list(set(subjects_total) - set(subjects_selected))
    
    #- set the seed based on <config.py>;
    random.seed(seed)
    return random.choice(subjects_remaining)


# select 10 random images from {select_mock};
def select_images(subject_id, dir_PIE, count=10):
    folder = os.path.join(dir_PIE, str(subject_id))
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(images) < count:
        raise ValueError(f"Not enough images in {folder} to select {count}. Found only {len(images)} ...")
    
    return [os.path.join(folder, f) for f in random.sample(images, count)]


# save the randomly selected subjects + images into a <.txt>;
def save_selection(subjects_selected, selected_images, dir_results, dir_database):
    dir_subjects = os.path.join(dir_results, "selected_subjects.txt")
    with open(dir_subjects, "w") as f:
        for sid in subjects_selected:
            f.write(f"{sid}\n")
    
    dir_images = os.path.join(dir_results, "selected_images.txt")
    with open(dir_images, "w") as f: 
        for img in selected_images:
            f.write(f"{os.path.relpath(img, start=dir_database)}\n")
    
    print_with_plus(f"saved selected subjects to {dir_subjects}")
    print_with_plus(f"saved the 10 randomly selected images to {dir_images}")


# plot the 10 randomly selected images for report;
def plot_selected(dir_images, subject_id, dir_save=None): 
    if len(dir_images) != 10: 
        raise ValueError("required exact number (10) to plot the 2x5 layout")
    
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Randomly Selected Images from Subject {subject_id}", fontsize=16)
    
    for i, dir_img in enumerate(dir_images):
        row, col = divmod(i, 5)
        axs[row, col].imshow(mping.imread(dir_img))
        axs[row, col].axis('off')
        axs[row,col].set_title(os.path.basename(dir_img), fontsize=8)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if dir_save:
        plt.savefig(dir_save, bbox_inches="tight", dpi=300)
        print_with_plus(f"array of images saved to {dir_save}")
        
    plt.show()
    

# split selected images into 70/30 for training and testing;
def split_7030(subject_id, dir_PIE, ratio_train=0.7, seed=seed):
    #- def folder to find the images -- based on <config.py>;
    folder = os.path.join(dir_PIE, str(subject_id))
    images_all = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    
    #- use random to split the images; 
    random.seed(seed)
    random.shuffle(images_all)
    
    #- assign the split ratio;
    #-- calculates no. of images -- use round cause python is stupid;
    idx_split = round(len(images_all) * ratio_train)
    dir_train = [(img, subject_id) for img in images_all[:idx_split]]
    dir_test = [(img, subject_id) for img in images_all[idx_split:]]
    return dir_train, dir_test


# save split data into <.csv>;
def save_split(label_images, file_output, dir_results):
    with open(file_output, "w") as f:
        for path, label in label_images:
            dir_rel = os.path.relpath(path, start=dir_results)
            f.write(f"{dir_rel}, {label}\n") 
    

if __name__=="__main__":
    # select 25 subjects out of the 68; 
    subjects_total= list(range(1, 69))
    subjects_selected = select_main(seed, subjects_total=68, sample_size=25)
    print(f"the 25 subjects selected are: {subjects_selected}")
    
    # select one random subject that is not part of the 25;
    subjects_mock = select_mock(subjects_total, subjects_selected)
    print(f"the mock subject selected is: {subjects_mock}")
    
    # select 10 images randomly from a random subject that is not part of the 25;
    selected_images = select_images(subjects_mock, dir_PIE)
    print(f"the 10 random images selected are from subject {subjects_mock}: ")
    for img in selected_images:
        print(f" - {os.path.relpath(img, start=dir_database)}")

    # save the screen output into a <.txt> file;
    save_selection(subjects_selected, selected_images, dir_thisPart, dir_database)
    
    # create name for layout of the 10 selected images; 
    dir_savedImages = os.path.join(dir_thisPart, f"subject{subjects_mock}_grid.png")
    # plot the 10 selected images; 
    plot_selected(selected_images, subjects_mock, dir_save=dir_savedImages)
    
    
    # check if every subject has 170 images ; 
    print_with_plus("Checking number of images in each selected subject folder:")
    for sid in subjects_selected:
        folder = os.path.join(dir_PIE, str(sid))
        all_images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Subject {sid}: {len(all_images)} images")
        

    # split the selected images into training and testing sets;
    images_train = []
    images_test = [] 
    
    # for PIE subjects;
    print_with_plus("Checking each subject split:") 
    for sid in subjects_selected: 
        train, test = split_7030(sid, dir_PIE, seed=seed)
        print(f"Subject {sid}: train = {len(train)}, test = {len(test)}, total = {len(train) + len(test)}")
        images_train.extend(train)
        images_test.extend(test)
    
    # do the split_7030 for the 10 imgs -- to retain the img file dir in the 25 subjects; 
    random.seed(seed)
    random.shuffle(selected_images)
    label_mock = max(subjects_selected) +1
    dir_train_mock = [(img, label_mock) for img in selected_images[:7]]
    dir_test_mock = [(img, label_mock) for img in selected_images[7:]]
    
    images_train.extend(dir_train_mock)
    images_test.extend(dir_test_mock)
    
    # save the overall split data into <.csv>; 
    save_split(images_train, os.path.join(dir_thisPart, "images_train.csv"), dir_thisPart)
    save_split(images_test, os.path.join(dir_thisPart, "images_test.csv"), dir_thisPart)
    
    print_with_plus(f"total training images: {len(images_train)}")
    print_with_plus(f"total testing images: {len(images_test)}")
    
