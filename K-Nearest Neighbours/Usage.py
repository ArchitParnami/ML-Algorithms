#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:10:46 2017

@author: archit
"""
from KNNClassifier import KNNClassifier

def Test():    
    X = [[1,0,'B'], [1,2, 'B'], [2,2,'A'], [2,3,'B'], [3,2,'A']]
    P = [2,1]
    K = 2
    #distance - 'euclidean', 'manhattan', 'hamming' 
    clf = KNNClassifier(X, K, distance='manhattan',weighted=True)
    out = clf.query(P)
    print(out)    
    
    
Test()