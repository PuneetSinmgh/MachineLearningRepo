# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:04:58 2018

@author: PuneetPC
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.naive_bayes import BernoulliNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")
for i in range(0,2):
    x_train, x_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1./3, random_state=3330)


features=len(x_train)

objects=len(y_train)


clf = BernoulliNB(alpha=0, binarize=0.0 , class_prior=None , fit_prior=True  )

clf = clf.fit(x_train, y_train)

a= clf._joint_log_likelihood(x_train)

print("joint log likelyhood train")

print(a)

res=[]
for i in range(0,objects):
    j=0
    res.append([a[i][j]/(a[i][j]+a[i][j+1]),a[i][j+1]/(a[i][j]+a[i][j+1])])
sum=0
for i in range(0,objects):
    if y_train[1].__eq__(False):
        sum += res[i][0]
    else:
        sum += res[i][1];

print("result...")
print(res) 
       
print("sum of log likelyhood...")
print(sum)        




objects=len(y_test)

a= clf._joint_log_likelihood(x_test)

print("joint log likelyhood test")

#print(a)

res=[]
for i in range(0,objects):
    j=0
    res.append([a[i][j]/(a[i][j]+a[i][j+1]),a[i][j+1]/(a[i][j]+a[i][j+1])])
sum=0
for i in range(0,objects):
    if y_train[1].__eq__(False):
        sum += res[i][0]
    else:
        sum += res[i][1];

print("result...")
print(res) 
       
print("sum of log likelyhood...")
print(sum) 

        