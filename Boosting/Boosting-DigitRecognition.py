#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:33:40 2017

@author: archit
"""


import pandas
import os
import matplotlib.pyplot as plt
import math
from sklearn import tree
import random


currentDirectory = os.getcwd()
path = currentDirectory + "/DigitRecognition/optdigits_raining.csv"
pathTest = currentDirectory + "/DigitRecognition/optdigits_test.csv"


def read_data(path):
    dataset = pandas.read_csv(path)
    array = dataset.values
    X = array[:, 0:64]
    Y = array[:, 64]
    return (X,Y)

def get_random_data(X, Y, count):
    indices = []
    total_items = len(X)
    
    while(len(indices) != count):
        r = random.randrange(total_items)
        if r not in indices:
            indices.append(r)
    
    rX = []
    rY = []
    
    for i in indices:
        rX.append(X[i])
        rY.append(Y[i])
        
    return (rX, rY)

def Train(X, Y, n_items, nIters, alphas, weights, weak_learners, errors, K):
    
    for i in range(nIters):
        
        hypothesis = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=2)
        hypothesis.fit(X, Y, sample_weight=weights)
        pred = hypothesis.predict(X)
        
        error = 0
        #norm = sum(weights)
        for j in range(n_items):
            if(pred[j] != Y[j]):
                error += weights[j]
                
        #error = error/ norm
        
        alpha = math.log((1-error)/error) + math.log(K-1)
                        
        for j in range(n_items):
            if(pred[j] == Y[j]):
                weights[j] = weights[j] / (K * (1-error))
            else:
                weights[j] = weights[j] / (K * error)
        
        weak_learners.append(hypothesis)
        alphas.append(alpha)
        errors.append(error)
 
def final_hypothesis_output(input, weak_learners, alphas, nIters):
    predVal = {}
    
    for i in range(nIters):
        pred = weak_learners[i].predict([input])
        pClass = pred[0]
        if pClass in predVal:
            predVal[pClass] += alphas[i]
        else:
            predVal[pClass] = alphas[i]
    
    result = -1
    maxVal = -math.inf
    for key, val in predVal.items():
        if val > maxVal:
            result = key
            maxVal = val
    
    return result
    

def Predict(input, weak_learners, alphas, nIters):
    pred = []
    for x in input:
        pred.append(final_hypothesis_output(x, weak_learners, alphas, nIters))
    return pred

def calculate_accuracy(pred, target):
    l = len(pred)
    errors = 0
    for i in range(l):
        if(pred[i] != target[i]):
            errors += 1
    accuracy = l - errors
    return accuracy * 100 / l


def Boost(rX, rY, n_items, nIters, K):
   
    weak_learners = []
    alphas = []
    weights= [1/n_items] * n_items
    errors = []
            
    Train(rX, rY, n_items, nIters, alphas, weights, weak_learners, errors, K)
    
    #plt.scatter(errors, alphas)
    #plt.show()
    
    pred = Predict(rX, weak_learners, alphas, nIters)
    acc = calculate_accuracy(pred, rY)
    return acc


def Run(n_items, start, maxItr, step):
    random.seed(42)
    Xi, Yi = read_data(path)
    rX, rY = get_random_data(Xi, Yi, n_items)
    output_classes = 10
    
    accuracy_rates = []
    
    for nIters in range(start, maxItr + 1, step):
        acc = Boost(rX, rY, n_items, nIters, output_classes)
        accuracy_rates.append(acc)
        
        print("Number of classifiers = ", nIters, "Accuracy = ", acc)
        
    plt.scatter(range(start, maxItr + 1, step), accuracy_rates)
    plt.show()
    
Run(200, 0, 60, 5)

    