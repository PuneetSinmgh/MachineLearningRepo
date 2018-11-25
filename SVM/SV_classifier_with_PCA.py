# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:56:43 2018

@author: PuneetPC
"""

import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

x_image_dataset = np.genfromtxt("../data/x.csv", delimiter=" ", dtype=np.float64)
#print(x_image_dataset)


y_image_labels = np.genfromtxt("../data/y_williams_vs_others.csv")

#print(y_image_labels)
#print(y_image_labels.shape)

print("willaims dataset")
print(x_image_dataset.shape)
print(" n component =3070, svd solver= auto")
pca = PCA(n_components=3070, svd_solver='auto' )

pca.fit(x_image_dataset)
    
X_new = pca.transform(x_image_dataset)

print(X_new.shape)

#param_grid = [
#  {'C': [ 10, 100,1000], 'kernel': ['rbf']},
#  {'C': [ 10, 100,1000], 'degree':[3], 'kernel': ['poly']},
#]

print(" C=1000.0, kernel=rbf , gamma=auto, degree=3 ")
svc = SVC(C=1000.0,kernel='rbf',cache_size=500, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

#clf = GridSearchCV(svc, param_grid)

#clf.fit(x_image_dataset,y_image_labels)

stratified_cv_results = cross_validate(svc, X_new, y_image_labels, 
                                       cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=3330),
                                       scoring=('precision', 'recall', 'f1'), 
                                      return_train_score=False)

#print(clf.cv_results_.keys())
#print("best parameter setting")
#print(clf.best_params_)

print(stratified_cv_results['test_f1'])
print("mean F1")
print(np.sum(stratified_cv_results['test_f1'])/3)

#print(stratified_cv_results['test_precision'])
#print(stratified_cv_results['test_recall'])