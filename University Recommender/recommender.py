import sys
import os
import collections
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn import svm
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from numpy.random import permutation
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
def euclideanDistance(data1, data2, length):
    distance=0
    for x in range(length):
        distance += (data1[x] - data2[x])**2
    return (distance)**0.5
def knn(trainingSet, testInstance, k):
    print(k)
    distances = {}
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)   
        distances[x] = dist
    sorted_d = sorted(distances.items(), key=lambda x: x[1]) 
    neighbors = []
    for x in range(k):
        neighbors.append(sorted_d[x][0])
    classVotes = {}
    for x in range(len(neighbors)):
        response = trainingSet[neighbors[x]][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
    return(sortedVotes, neighbors)
def recommend(mark2):
    final_data = pd.read_csv("Rec_data.csv")
    result,neigh= knn(final_data.values.tolist(), mark2[:10], mark2[10])
    list1 = []
    list2 = []
    for i in result:
        list1.append(i[0])
        list2.append(i[1])
    return list1