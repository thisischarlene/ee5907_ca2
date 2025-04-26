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
    print("part1_1: randomly choosing 25 of the 68 subjects from provided database ...")
    A0280349Y.part1_1.main()
    
    print("\npart1_2a: implement PCA from scratch ...")
    A0280349Y.part1_2a.main()
    print("\npart1_2b: apply crafted PCA function to training set with number of PCs to be 2 and 3 ...")
    A0280349Y.part1_2b.main()
    print("\npart1_2c: apply crafted PCA function to training set with number of PCs to be 80 and 200 ...")
    A0280349Y.part1_2c.main()
    
   
    print("\npart1_3a: implement LDA from scratch ...")
    A0280349Y.part1_3a.main()
    print("\npart1_3b: apply crafted LDA function to training set with number of LDA projection vectors to be 2 and 3...")
    A0280349Y.part1_3b.main()
    
    print("\npart1_3c: apply crafted LDA function to training set with number of LDA projection vectors to be 9 and 15 ...")
    A0280349Y.part1_3c.main()
    print_with_star("all scripts for part1 have been executed ... ")
    
  
    # part2 of assignment2;
    print("\npart2_1a: apply KNN to PCA transformed testing dataset for p=80 and p=200 with k=1 ...")
    A0280349Y.part2_1a.main()
    print("\npart2_1b: apply KNN to PCA transformed testing dataset for p=9 and p=15 with k=1 ...")
    A0280349Y.part2_1b.main()
    
    print("\npart2_2a: using raw vectorized training images to train GMM model with 3 Gaussian components...")
    A0280349Y.part2_2a.main()
    print("\npart2_2b: use PCA transformed training images to train GMM model with 3 Gaussian components for  p=80 and p=200...")
    A0280349Y.part2_2b.main()
    
    print("\npart2_3a: apply linear SVM to raw vectorized face images ...")
    A0280349Y.part2_3a.main()
    print("\npart2_3b: apply linear SVM to PCA transformed feature representations for p=80 and p=200 ...")
    A0280349Y.part2_3b.main()
    print("\npart2_3c: apply linear SVM to LDA transformed feature representation for p=9 and p=15 ...")
    A0280349Y.part2_3c.main()
   
    print("\npart2_4:use raw training face images to train CNN with 2 convolutional layers ...")
    A0280349Y.part2_4.main()
    print_with_star("all scripts for part2 have been executed ... ")
    

if __name__ == "__main__":
    main()