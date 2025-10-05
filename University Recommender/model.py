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
# from hummingbird.ml import convert
from sklearn.ensemble import AdaBoostClassifier
import json
from xgboost import XGBClassifier

def result_prediction(mark):
    if mark[7]!=-1:
        xgb = XGBClassifier()
        xgb.load_model('xgboost_model_sub.bin')    
    else:
        xgb = XGBClassifier()
        xgb.load_model('xgboost_model.bin')        
        mark.remove(-1)
    # Assuming you have a JSON file named 'data.json'
    with open('univ_dic.json', 'r') as file:
        univ_dic = json.load(file)
    with open('stream_dic.json', 'r') as file:
        stream_dic = json.load(file)
    with open('program_dic.json', 'r') as file:
        program_dic = json.load(file)
    with open('citi_dic.json', 'r') as file:
        citi_dic = json.load(file)
    mark[0]=univ_dic[mark[0]]
    mark[1]=stream_dic[mark[1]]
    mark[2]=program_dic[mark[2]]
    mark[-1]=citi_dic[mark[-1]]
# Load the saved model
    
    
    prediction_xgb=xgb.predict([mark])
    return prediction_xgb[0]