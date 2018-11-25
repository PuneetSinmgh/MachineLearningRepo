# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:18:23 2018

@author: PuneetPC
"""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

#file = open('../data/x.csv','r')
#lines = list()
#x_image_dataset= np.array([])
#for l in file:
 #  lines = lines.append(l)
   #line = np.asarray(line, dtype= np.float64)   

x_image_dataset = np.genfromtxt("../data/x.csv", delimiter=" ", dtype=np.float64)



y_image_labels = np.genfromtxt("../data/y_williams_vs_others.csv")


KNN_clf = KNeighborsClassifier(n_neighbors=3)

stratified_cv_results = cross_validate(KNN_clf, x_image_dataset, y_image_labels, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=3330),
                                       scoring=('precision', 'recall', 'f1'), 
                                       return_train_score=False)


print(stratified_cv_results['fit_time'])
print(stratified_cv_results['score_time'])
print(stratified_cv_results['test_f1'])
print(stratified_cv_results['test_precision'])
print(stratified_cv_results['test_recall'])
