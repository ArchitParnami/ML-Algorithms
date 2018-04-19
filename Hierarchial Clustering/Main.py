# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:24:56 2017

@author: Archit
"""

from HierarchialClustering import HierarchialClustering
import csv
from scipy.cluster.hierarchy import linkage
from scipy.cluster import hierarchy

import matplotlib.pyplot as plt


def read_data(filename):
    X = []
    Y = []
    
    with open(filename, 'r') as df:
        reader = csv.reader(df)
        skipOne = True
        for row in reader:
            if not skipOne:
                Y.append(row[0])
                x = [float(item) for item in row[1:]]
                X.append(x)
            else:
                skipOne = False
    return (X,Y)
    

filename = "SCLC_study_output_filtered_2.csv"

X, Y = read_data(filename) 

#My implementation for Hierarchial Clustering
H = HierarchialClustering()
Z = H.fit(X, Y)
out = H.output();
print("Output Format: (cluster_number, cluster-A, cluster-B)\n")
print(out)

#use scipy library to verify the results
Z = linkage(X)
plt.figure()
dn = hierarchy.dendrogram(Z, labels=Y, orientation='right')


    