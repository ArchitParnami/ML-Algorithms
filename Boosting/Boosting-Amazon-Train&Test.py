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
path = currentDirectory + "/Train-SentimentAnalysis.csv"
pathTest = currentDirectory + "/Test-SentimentAnalysis.csv"


def read_data(path):
    dataset = pandas.read_csv(path)
    array = dataset.values
    X = array[1:, 0:2]
    Y = array[1:, 2]
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

def Train(X, Y, n_items, nIters, alphas, weights, weak_learners, errors):
    
    for i in range(nIters):
        
        hypothesis = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=1)
        hypothesis.fit(X, Y, sample_weight=weights)
        pred = hypothesis.predict(X)
        
        error = 0
        norm = sum(weights)
        for j in range(n_items):
            if(pred[j] != Y[j]):
                error += weights[j]
                
        error = error/ norm
        
        alpha = math.log((1-error)/error) / 2
                        
        for j in range(n_items):
            if(pred[j] == Y[j]):
                weights[j] = weights[j] / (2 * (1-error))
            else:
                weights[j] = weights[j] / (2 * error)
        
        weak_learners.append(hypothesis)
        alphas.append(alpha)
        errors.append(error)
 
def final_hypothesis_output(input, weak_learners, alphas, nIters):
    sum = 0
    
    for i in range(nIters):
        pred = weak_learners[i].predict([input])
        sum += alphas[i] * pred[0]
    return 1 if sum > 0 else -1

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


def Boost(rX, rY, n_items, nIters, rXt, rYt):
   
    weak_learners = []
    alphas = []
    weights= [1/n_items] * n_items
    errors = []
            
    Train(rX, rY, n_items, nIters, alphas, weights, weak_learners, errors)
    
    #plt.scatter(errors, alphas)
    #plt.show()
    
    pred_train = Predict(rX, weak_learners, alphas, nIters)
    pred_test = Predict(rXt, weak_learners, alphas, nIters)
    acc_train = calculate_accuracy(pred_train, rY)
    acc_test = calculate_accuracy(pred_test, rYt)
    return (acc_train, acc_test)


def Run(n_items, start, maxItr, step):
    random.seed(42)
    Xi, Yi = read_data(path)
    rX, rY = get_random_data(Xi, Yi, n_items)
    
    Xt, Yt = read_data(pathTest)
    rXt, rYt = get_random_data(Xt, Yt, n_items)
    
    accuracy_rates_train = []
    accuracy_rates_test = []
    
    for nIters in range(start, maxItr + 1, step):
        acc_train, acc_test = Boost(rX, rY, n_items, nIters, rXt, rYt)
        accuracy_rates_train.append(acc_train)
        accuracy_rates_test.append(acc_test)
        
        print("Number of classifiers = ", nIters, "Training Accuracy = ", acc_train, "Test Accuracy = ", acc_test)
        
    plt.scatter(range(start, maxItr + 1, step), accuracy_rates_train)
    plt.show()
    plt.scatter(range(start, maxItr + 1, step), accuracy_rates_test)
    plt.show()
    
Run(100, 20, 260, 20)

    