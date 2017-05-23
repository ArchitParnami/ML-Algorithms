#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:00:11 2017

@author: archit
"""

class BayesClassifier(object):
    def __init__(self):
        self.number_of_examples = None
        self.number_of_classes = None
        self.number_of_attributes = None
        self.occurances = None
        
    def __initialize(self, X, Y):
        self.number_of_examples = len(X)
        self.number_of_attributes = len(X[0])
        output_classes = []
        for y in Y:
            if y not in output_classes:
                output_classes.append(y)
        self.number_of_classes = len(output_classes)
        
    
    def __classify_data(self, X, Y):
        ''' return {class1 : [rows], class2: [rows]}'''
        classified_data = {}
        for i in range(self.number_of_examples):
            out_class = Y[i]
            if out_class not in classified_data:
                classified_data[out_class] = []
            classified_data[out_class].append(X[i])
        return classified_data

    def __calculate_occurances(self, classified_data):
        occurances = {}
    
        for out_class, data_rows in classified_data.items():    
            attr_dict = {}
    
            for row in data_rows:
                attr_no = 1    
                for val in row:
                    
                    if attr_no not in attr_dict:
                        # {1 : {}}
                        attr_dict[attr_no] = {}
                    
                    val_dict = attr_dict[attr_no]
                    
                    #{2:10, 1:3}
                    if val not in val_dict:
                        val_dict[val] = 1
                    else:
                        val_dict[val] += 1
                                    
                    attr_no += 1
            
            occurances[out_class] = (len(data_rows), attr_dict)
    
        return occurances
    
    
    def __calculate_probability(self, x):
        prob = []

        for out_class, (n, attr_dict) in self.occurances.items():
            total_prob = 1
            for i in range(self.number_of_attributes):
              val_dict = attr_dict[i+1]
              val_count = 0
              num_of_unique_keys = len(val_dict.keys())
              if x[i] in val_dict:
                  val_count = val_dict[x[i]]
              individual_prob = (val_count + 1) / (n + num_of_unique_keys)
              total_prob *= individual_prob
            
            prior = n / self.number_of_examples
            total_prob *= prior
            prob.append((out_class, total_prob))
        
        return prob
    
    def __get_class_with_max_prob(self, prob):
        
        pmax = 0
        c = None
        
        for out_class, p in prob:
            if p > pmax:
                pmax = p
                c = out_class
        
        return c
    
    def __classify(self, x):
        probs = self.__calculate_probability(x)
        return self.__get_class_with_max_prob(probs)
    
    def fit(self, X, Y):
        self.__initialize(X, Y)
        classified_data = self.__classify_data(X, Y)
        self.occurances = self.__calculate_occurances(classified_data)
        
    
    def predict(self, X):
        predictions = []
        for x in X:
            c = self.__classify(x)
            predictions.append(c)
        return predictions
        
    