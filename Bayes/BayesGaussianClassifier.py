#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:00:11 2017

@author: archit
"""
import math

class BayesGaussianClassifier(object):
    def __init__(self):
        self.number_of_examples = None
        self.number_of_classes = None
        self.number_of_attributes = None
        self.means = None
        self.stddevs = None
        self.output_classes = None
        self.classified_data = None
        
    def __initialize(self, X, Y):
        self.number_of_examples = len(X)
        self.number_of_attributes = len(X[0])
        output_classes = []
        for y in Y:
            if y not in output_classes:
                output_classes.append(y)
        self.number_of_classes = len(output_classes)
        self.output_classes = output_classes
    
    def __classify_data(self, X, Y):
        ''' return {class1 : [rows], class2: [rows]}'''
        classified_data = {}
        for i in range(self.number_of_examples):
            out_class = Y[i]
            if out_class not in classified_data:
                classified_data[out_class] = []
            classified_data[out_class].append(X[i])
        return classified_data

    
    def __calculate_mean(self, classified_data):
        means = {}
        for out_class, data_rows in classified_data.items():
            n = len(data_rows)
            attr_dict = {}
            for row in data_rows:
                attr_no = 1
                for val in row:
                    if attr_no not in attr_dict:
                        attr_dict[attr_no] = 0
                    attr_dict[attr_no] += val
                    attr_no += 1
            
            for i in range(1, self.number_of_attributes+1):
                attr_dict[i] = attr_dict[i] / n
            
            means[out_class] = attr_dict
                 
        return means
    
    def __calculate_stddev(self, classified_data, means):
         stddev = {}
         for out_class, data_rows in classified_data.items():
             n = len(data_rows)
             attr_dict = {}
             for row in data_rows:
                 attr_no = 1
                 for val in row:
                     if attr_no not in attr_dict:
                         attr_dict[attr_no] = 0
                     mean = means[out_class][attr_no]
                     attr_dict[attr_no] += math.pow(val - mean, 2)
                     attr_no += 1
            
             for i in range(1, self.number_of_attributes+1):
                 attr_dict[i] = math.sqrt(attr_dict[i] / (n-1))
            
             stddev[out_class] = attr_dict
                  
         return stddev
    
    def __calculate_gaussian_probability(self, X):
        probabs = []
        
        for out_class in self.output_classes:
            attr_no = 1
            prob = 1
            for x in X:
                mean = self.means[out_class][attr_no]
                stdev = self.stddevs[out_class][attr_no]
                exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
                p =  (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
                prob *= p
            
            prior = len(self.classified_data[out_class]) / self.number_of_examples
            prob = prior * prob            
            probabs.append((out_class, prob))
         
        return probabs       
    
    def __get_class_with_max_prob(self, prob):
        
        pmax = 0
        c = None
        
        for out_class, p in prob:
            if p > pmax:
                pmax = p
                c = out_class
        
        return c
    
    def __classify(self, x):
        probs = self.__calculate_gaussian_probability(x)
        return self.__get_class_with_max_prob(probs)
    
    def fit(self, X, Y):
        self.__initialize(X, Y)
        self.classified_data = self.__classify_data(X, Y)
        self.means = self.__calculate_mean(self.classified_data)
        self.stddevs = self.__calculate_stddev(self.classified_data, self.means)
    
    def predict(self, X):
        predictions = []
        for x in X:
            c = self.__classify(x)
            predictions.append(c)
        return predictions
        
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 12:43:32 2017

@author: archit
"""

