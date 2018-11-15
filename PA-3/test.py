# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:23:44 2018

@author: PuneetPC
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")


l2_model_complexity = np.zeros((10, 15))
l2_train_cll = np.zeros((10, 15))
l2_test_cll = np.zeros((10, 15))

l2_num_zero_weights = np.zeros((10,15))
l1_num_zero_weights = np.zeros((10,15))
## todo 1
complexity= [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7]

for i in range(0,2):
    x_train, x_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1./3, random_state=3330)

samples_train=len(y_train)
samples_test= len(y_test)
clfL2 = LogisticRegression(penalty='l2',C=complexity[11],random_state=42)

clfL2.fit(x_train, y_train)

log_prob=[]
log_prob=clfL2.predict_log_proba(x_train)

mod_com=0
coefl2=[]
coefl2 = clfL2.coef_
for i in range(0,len(coefl2)):
    for j in range(0,len(coefl2[i])):
        mod_com += np.square(coefl2[0][j])
print(mod_com)

#print(coefl2)

l2_zero_count=0
for i in range(0,len(coefl2)):
    for j in range(0,len(coefl2[i])):
        if coefl2[0][j] == 0:
            l2_zero_count +=1
print("zero count")
print(l2_zero_count)

sum1=0
for k in range(0,samples_train):
    if y_train[k].__eq__(False):
        sum1 += log_prob[k][0]
    else:
        sum1 += log_prob[k][1]

l2_train_cll = sum1



log_prob=[]
log_prob=clfL2.predict_log_proba(x_test)

sum2=0
for k in range(0,samples_test):
    if y_test[k].__eq__(False):
        sum2 += log_prob[k][0]
    else:
        sum2 += log_prob[k][1]

l2_test_cll = sum2


# l1 classifier  
samples_train=len(y_train)
clfL1 = LogisticRegression(penalty='l1',C=complexity[11],random_state=42)

clfL1.fit(x_train, y_train)

coefl1=[]
coefl1 = clfL1.coef_

l1_zero_count=0
for i in range(0,len(coefl1)):
    for j in range(0,len(coefl1[i])):
        if coefl1[0][j] == 0:
            l1_zero_count +=1

print(l1_zero_count)