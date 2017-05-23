#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: archit
"""

import math
import operator

class KNNClassifier():
    def __init__(self, X, K, distance='euclidean', weighted=False):
        self.input = X
        self.K = K
        self.distance_f = self.__get_distance_function(distance)
        self.weighted = weighted
    
    def __eulidean_distance(self, x, q):
        l = len(x) - 1
        distance = 0
        for i in range(l):
            distance += pow((x[i] - q[i]), 2)
        return math.sqrt(distance)
    
    def __manhattan_distance(self, x, q):
        l = len(x) - 1
        distance = 0
        for i in range(l):
            distance += abs(x[i] - q[i])
        return distance
    
    def __hamming_distance(self, x, q):
        return sum(el1 != el2 for el1, el2 in zip(x[:-1], q))
    
    def __get_distance_function(self, name):
        function_dict = {
                'euclidean' : self.__eulidean_distance,
                'manhattan' : self.__manhattan_distance,
                'hamming'   : self.__hamming_distance
                }
        
        return function_dict[name]
    
    def __get_nearest_neighbours(self, query_point):
        distances  = [ (x, self.distance_f(x, query_point)) for x in self.input]
        distances.sort(key=operator.itemgetter(1))
        uniques = 0
        nbrs = []
        for dp in distances:
            if (len(nbrs) == 0 or dp[1] != nbrs[-1][1]):
                uniques += 1
                
            if uniques > self.K:
                break;
            
            nbrs.append(dp)
            
        return nbrs;
    
    def __find_majority(self, neighbours):
        
        if neighbours[0][1] == 0:
            return neighbours[0][0][-1]
        
        l = len(neighbours)
        votes = {}
        for i in range(l):
            y = neighbours[i][0][-1]
            
            weight = 1/neighbours[i][1]**2 if self.weighted else 1
            
            if y in votes:
                votes[y] += weight
            else:
                votes[y] = weight
                     
        sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        #print(sortedVotes)
        return sortedVotes[0][0]
    
    def query(self, P):
        neighbours = self.__get_nearest_neighbours(P)
        #print(neighbours)
        return self.__find_majority(neighbours)
    

    
    
        
        
        
    
    
    