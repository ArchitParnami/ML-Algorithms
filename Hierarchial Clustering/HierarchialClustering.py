# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:35:07 2017

@author: Archit
"""

import math

class Cluster:
    def __init__(self):
        self.data = []
        self.label = None
        self.left_child = None
        self.right_child = None
    
class ClusterSimilarityCalculator:
        def __init__(self, linkage, distance):
            
            if linkage.lower() == "single":
                self.similarity = self.__find_closest_distance
            
            if distance.lower() == "euclidean":
                self.distance = self.__euclidean_distance
        
        def calculate(self, cluster_1, cluster_2):
            return self.similarity(cluster_1, cluster_2)
        
        def __find_closest_distance(self, cluster_1, cluster_2):
            minimum = float('+inf')
            for x1 in cluster_1.data:
                for x2 in cluster_2.data:
                    d = self.distance(x1, x2)
                    if d < minimum:
                        minimum = d
            return minimum
                
        def __euclidean_distance(self, x1, x2):
            n = len(x1)
            dist = 0
            for i in range(n):
                dist += (x1[i] - x2[i]) ** 2
            
            return math.sqrt(dist)
    


class HierarchialClustering:
    def __init__(self, linkage='single', distance='euclidean'):
        self.X = []
        self.Y = []
        self.maxs = []
        self.mins = []
        self.XNorm = []
        self.clusters = []
        self.similarityCalcualtor = ClusterSimilarityCalculator(linkage, distance)
        
        
    def fit(self, X, Y, normalize=False):
        self.X = X
        self.Y = Y
        if normalize:
            self.__normalize()
        else:
            self.XNorm = self.X;
        self.__form_clusters()
        self.__start_clustering()
     
    
    def __normalize(self):
        self.__find_max_mins()
        
        rows = len(self.X)
        cols = len(self.X[0])
        
        for i in range(rows):
            x = []
            for j in range(cols):
              inorm = (self.X[i][j] - self.mins[j]) / (self.maxs[j] - self.mins[j])
              x.append(inorm)
            self.XNorm.append(x)
    
    
    def __find_max_mins(self):
        rows = len(self.X)
        cols = len(self.X[0])
       
        for i in range(cols):
           maximum = float('-inf')
           minimum = float('+inf')
           for j in range(rows):
               item = self.X[j][i]
               if item > maximum:
                   maximum = item
               if item < minimum:
                   minimum = item
           self.maxs.append(maximum)
           self.mins.append(minimum)
           
           
    def __form_clusters(self):
        n = len(self.XNorm)
        for i in range(n):
            cluster = Cluster()
            cluster.data = [self.XNorm[i]]
            cluster.label = self.Y[i]
            cluster.left_child = None
            cluster.right_child = None
            self.clusters.append(cluster)
            
    def __start_clustering(self):
        count = 1;
        while(len(self.clusters) > 1):
            c1, c2 = self.__find_most_similar()
            c = Cluster()
            c.data = c1.data + c2.data
            c.left_child = c1
            c.right_child = c2
            c.label = (count, c1.label, c2.label)
            self.clusters.remove(c1)
            self.clusters.remove(c2)
            self.clusters.append(c)
            count += 1
    
    def __find_most_similar(self):
        minimum = float('+inf') 
        closest_pair = None
        n = len(self.clusters)
        for i in range(n):
            for j in range(i+1, n):
                c1 = self.clusters[i]
                c2 = self.clusters[j]
                score = self.similarityCalcualtor.calculate(c1, c2)
                if score < minimum:
                    minimum = score
                    closest_pair = (c1, c2)
                    
        return closest_pair
    
    def output(self):
        if len(self.clusters) == 1:
            return self.clusters[0].label
    
    def root(self):
        if len(self.clusters) == 1:
            return self.clusters[0]
                        