# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:12:15 2018

@author: PuneetPC
"""
#import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")


l2_model_complexity = np.zeros((10, 15))
l2_train_cll = np.zeros((10, 15))
l2_test_cll = np.zeros((10, 15))

l2_num_zero_weights = np.zeros((10,15))
l1_num_zero_weights = np.zeros((10,15))
## complecity or the value of C
complexity= [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7]
# exponent of C
expofC = np.exp(complexity)
for i in range(0,10):
    x_train, x_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1./3, random_state=3330)

    samples_train=len(y_train)
   
    samples_test= len(y_test)
    
    for j in range(0,15):
        
        clfL2 = LogisticRegression(penalty='l2',C=complexity[j],random_state=42)

        clfL2.fit(x_train, y_train)

        #model complexity
        mod_com=0
        coefl2=[]
        coefl2 = clfL2.coef_
        for k in range(0,len(coefl2)):
            for l in range(0,len(coefl2[k])):
                mod_com += np.square(coefl2[k][l])
                
        l2_model_complexity[i][j]=mod_com

        #number of l2 zero weights
        l2_zero_count=0
        for k in range(0,len(coefl2)):
            for l in range(0,len(coefl2[k])):
                if coefl2[k][l] == 0:
                    l2_zero_count +=1

        l2_num_zero_weights[i][j]=l2_zero_count

        #CLL train data set L2 model complexity

        log_prob=[]
        log_prob=clfL2.predict_log_proba(x_train)

        sum1=0
        for k in range(0,samples_train):
            if y_train[k].__eq__(False):
                sum1 += log_prob[k][0]
            else:
                sum1 += log_prob[k][1]

        l2_train_cll[i][j] = sum1

        #CLL tets data set L2 model complexity

        log_prob=[]
        log_prob=clfL2.predict_log_proba(x_test)

        sum2=0
        for k in range(0,samples_test):
            if y_test[k].__eq__(False):
                sum2 += log_prob[k][0]
            else:
                sum2 += log_prob[k][1]

        l2_test_cll[i][j] = sum2

        # Logistic regression classifier with L1 penalty

        clfL1 = LogisticRegression(penalty='l1',C=complexity[j],random_state=42)
        clfL1.fit(x_train, y_train)

        coefl1=[]
        coefl1 = clfL1.coef_

        l1_zero_count=0
        for k in range(0,len(coefl1)):
            for l in range(0,len(coefl1[k])):
                if coefl1[k][l] == 0:
                    l1_zero_count +=1

        l1_num_zero_weights[i][j]=l1_zero_count
        
    plt.plot(l2_model_complexity[i],l2_train_cll[i], linewidth=2, label='Train CLL')
    plt.plot(l2_model_complexity[i],l2_test_cll[i], linewidth=2, label='Test CLL')
    plt.ylabel('Train & Test CLL')
    plt.xlabel('Model Complexity')
    plt.title(i)
    plt.legend(loc='right')
    plt.show()
    
    plt.plot(expofC,l2_num_zero_weights[i], linewidth=2, label='l2 zero weights')
    plt.plot(expofC,l1_num_zero_weights[i], linewidth=2, label='l1 zero weights')
    plt.ylabel('No of Zero weights')
    plt.xlabel('Exponent of C')
    plt.title(i)
    plt.legend(loc='right')
    plt.show()
        

pickle.dump((l2_model_complexity, l2_train_cll, l2_test_cll, l2_num_zero_weights, l1_num_zero_weights), open('result.pkl', 'wb'))