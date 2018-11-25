# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 01:36:21 2018

@author: PuneetPC
"""
import pickle
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

#file = open('../data/x.csv','r')
#lines = list()
#x_image_dataset= np.array([])
#for l in file:
 #  lines = lines.append(l)
   #line = np.asarray(line, dtype= np.float64)   

x_image_dataset = np.genfromtxt("../data/x.csv", delimiter=" ", dtype=np.float64)

y_image_labels = np.genfromtxt("../data/y_williams_vs_others.csv")

print("willaims dataset")
print(x_image_dataset.shape)
print(" n component =50, svd solver= auto")
pca = PCA(n_components=50, svd_solver='auto' )

pca.fit(x_image_dataset)
    
X_new = pca.transform(x_image_dataset)

print(X_new.shape)
print("KNN = 5")
KNN_clf = KNeighborsClassifier(n_neighbors=5)

stratified_cv_results = cross_validate(KNN_clf, X_new, y_image_labels, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=3330),
                                       scoring=('precision', 'recall', 'f1'), 
                                       return_train_score=False)


#print(stratified_cv_results['fit_time'])
#print(stratified_cv_results['score_time'])
print(stratified_cv_results['test_f1'])

print("mean F1")
print(np.sum(stratified_cv_results['test_f1'])/3)

#print(stratified_cv_results['test_precision'])
#print(stratified_cv_results['test_recall'])

#pickle.dump((stratified_cv_results), open('KNN_5_Bush.pkl', 'wb'))