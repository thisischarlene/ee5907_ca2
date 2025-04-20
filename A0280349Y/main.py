#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<main.py> { 
    context         <.runAll> file; 
    purpose         runs all files in this package in proper sequence; 
    used in         parent folder [A0280349Y]; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<main.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""
import A0280349Y
from A0280349Y.config import *

def main():
    print("\n")
    print_with_star("running all scripts for EE5907 CA2 ...")
    
    # part1 of assignment2; 
    print("randomly choosing 25 of the 68 subjects from provided database ...")
    A0280349Y.part1_1.main()
    
    print("\nimplement PCA from scratch ...")
    A0280349Y.part1_2a.main()
    """
    print("\napply crafted PCA function to training set with number of PCs to be 2 and 3 ...")
    A0280349Y.part1_2b.main()
    print("\napply crafted PCA function to training set with number of PCs to be 80 and 200 ...")
    A0280349Y.part1_2c.main()
    
    print("\nimplement LDA from scratch ...")
    A0280349Y.part1_3a.main()
    print("\napply crafted LDA function to training set with number of LDA projection vectors to be 2 and 3...")
    A0280349Y.part1_3b.main()
    print("\napply crafted LDA function to training set with number of LDA projection vectors to be 9 and 15 ...")
    A0280349Y.part1_3c.main()
    print_with_star("all scripts for part1 have been executed ... ")
    
    
    # part2 of assignment2;
    print("\napply KNN to PCA transformed testing dataset for p=80 and p=200 with k=1 ...")
    A0280349Y.part2_1a.main()
    print("\napply KNN to PCA transformed testing dataset for p=9 and p=15 with k=1 ...")
    A0280349Y.part2_1b.main()
    
    print("\nusing raw vectorized training images to train GMM model with 3 Gaussian components...")
    A0280349Y.part2_2a.main()
    print("\nuse PCA transformed training images to train GMM model with 3 Gaussian components for  p=80 and p=200...")
    A0280349Y.part2_2b.main()
    
    print("\napply linear SVM to raw vectorized face images ...")
    A0280349Y.part2_3a.main()
    print("\napply linear SVM to PCA transformed feature representations for p=80 and p=200 ...")
    A0280349Y.part2_3b.main()
    print("\napply linear SVM to LDA transformed feature representation for p=9 and p=15 ...")
    A0280349Y.part2_3c.main()
    
    print("\nuse raw training face images to train CNN with 2 convolutional layers ...")
    A0280349Y.part2_4.main()
    print_with_star("all scripts for part2 have been executed ... ")
   
    
    """


if __name__ == "__main__":
    main()