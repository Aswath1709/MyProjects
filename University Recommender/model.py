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

def result_prediction(mark):
    df2=pd.read_csv("Prediction_data.csv")
    if mark[10]=="MS/PhD":
        df2 = df2[["greV","greQ","greA","cgpa","toeflScore","internExp","confPubs","journalPubs","researchExp","industryExp","admit"]][(df2["university"]==mark[11])&((df2["program"]=="MS")|(df2["program"]=="PhD")|(df2["program"]==mark[10]))]
    else:
        df2 = df2[["greV","greQ","greA","cgpa","toeflScore","internExp","confPubs","journalPubs","researchExp","industryExp","admit"]][(df2["university"]==mark[11])&(df2["program"]==mark[10])]
    df2.reset_index(inplace=True,drop=True)
    y= df2.admit
    x=df2.drop('admit',axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
    ab = AdaBoostClassifier(n_estimators=100, random_state=0)
    ab.fit(x_train, y_train)
    prediction_ab=ab.predict([mark[:10]])
    return prediction_ab[0]