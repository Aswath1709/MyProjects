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
import json
def euclideanDistance(data1, data2, length):
    print(data1)
    print(data2)
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
def recommend(mark2,k):
    if mark2[6]==-1:
        final_data=pd.read_csv("General.csv")
        mark2.remove(-1)
    
    else:
        final_data=pd.read_csv("Subject.csv")        
    final_data=final_data.drop(columns=["decision"],axis=1)
# Move the first column to the end
    
    with open('stream_dic.json', 'r') as file:
        stream_dic = json.load(file)
    with open('program_dic.json', 'r') as file:
        program_dic = json.load(file)
    with open('citi_dic.json', 'r') as file:
        citi_dic = json.load(file)
    mark2[0]=stream_dic[mark2[0]]
    mark2[1]=program_dic[mark2[1]]
    mark2[-1]=citi_dic[mark2[-1]]
    univ=final_data["university"]
    final_data=final_data.drop(columns=["university"])
    final_data=final_data.drop(columns=["Unnamed: 0"])
    final_data["university"]=univ
    final_data=final_data.values.tolist()
    print(mark2)
    print(final_data[2])
    result,neigh= knn(final_data, mark2, k)
    list1 = []
    list2 = []
    for i in result:
        list1.append(i[0])
        list2.append(i[1])
    with open('rev_univ_dic.json', 'r') as file:
        rev_univ_dic = json.load(file)
    return [rev_univ_dic[x] for x in list1 if x in rev_univ_dic.keys()]