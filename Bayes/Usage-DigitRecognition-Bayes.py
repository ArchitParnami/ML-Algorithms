#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:58:09 2017

@author: archit
"""
import pandas
import os
from BayesClassifier import BayesClassifier


currentDirectory = os.getcwd()
path = currentDirectory + "/DigitRecognition/optdigits_raining.csv"
pathTest = currentDirectory + "/DigitRecognition/optdigits_test.csv"

dataset = pandas.read_csv(path, header=None)
testDataset = pandas.read_csv(pathTest, header=None)

array = dataset.values
X = array[:, 0:64]
Y = array[:, 64]


testArray = testDataset.values
Xt = testArray[:, 0:64]
Yt = testArray[:, 64]


def Run(X, Y, Xt, Yt):
    bc = BayesClassifier()
    bc.fit(X, Y)
    pred = bc.predict(Xt)
    error = 0
    n = len(Xt)
    for i in range(n):
        if pred[i] != Yt[i]:
            error += 1
    
    print("Accuracy:", (n-error)/n * 100)
    
Run(X, Y, Xt, Yt)
