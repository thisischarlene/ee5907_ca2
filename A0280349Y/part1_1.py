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
import random
from A0280349Y.config import *
#from database.PIE import *

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

if __name__=="__main__":
    subjects_total= list(range(1, 69))
    subjects_selected = select_main(seed, subjects_total=68, sample_size=25)
    print_with_plus(f"the 25 subjects selected are: {subjects_selected}")
    
    mock_subject = select_mock(subjects_total, subjects_selected)
    print_with_plus(f"the mock subject selected is: {mock_subject}")
    
    mock_images = select_images(mock_subject, dir_PIE)
    print_with_plus(f"the 10 random images selected are from subject {mock_subject}: ")
    for img in mock_images:
        print(f" - {os.path.relpath(img, start=dir_database)}")