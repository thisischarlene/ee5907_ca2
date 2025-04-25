#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part2_1a.py> { 
    context         ; 
    purpose         applying KNN to PCA; 
    used in         parent folder [A0280349Y]; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<part2_1a.py>"
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
from sklearn.neighbors import


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_1] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part2_1a
os.makedirs(dir_thisPart, exist_ok=True)