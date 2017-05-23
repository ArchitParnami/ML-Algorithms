#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:20:36 2017

@author: archit
"""


import pandas
import os
from BayesGaussianClassifier import BayesGaussianClassifier

currentDirectory = os.getcwd()
path = currentDirectory + "/Train-SentimentAnalysis.csv"
pathTest = currentDirectory + "/Test-SentimentAnalysis.csv"


def read_data(path):
    dataset = pandas.read_csv(path)
    array = dataset.values
    X = array[1:, 0:2]
    Y = array[1:, 2]
    return (X,Y)


X, Y = read_data(path)
Xt, Yt = read_data(pathTest)

def Run(X, Y, Xt, Yt):
    bc = BayesGaussianClassifier()
    bc.fit(X, Y)
    pred = bc.predict(Xt)
    error = 0
    n = len(Xt)
    for i in range(n):
        if pred[i] != Yt[i]:
            error += 1
    
    print("Accuracy:", (n-error)/n * 100)
    
Run(X, Y, Xt, Yt)

