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
#from database.PIE import *


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