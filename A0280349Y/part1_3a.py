#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part1_3a.py> { 
    context         ee5907 assignment 2; 
    purpose         write custom LDA function; 
    used in         [A0280349Y];
    py version      3.10;
    os              windows 10; 
    ref(s)          ;       
    note            for master branch only; 
} // "<part1_3a.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""

import os
import datetime
import numpy as np 
import matplotlib.pyplot as plt
import random
from A0280349Y.config import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_2a] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part1_3a
os.makedirs(dir_thisPart, exist_ok=True)

def my_lda(X, y, p): 


def main():
    print("LDA function, {my_lda}, has been defined ...")

if __name__=="__main__":
    main()