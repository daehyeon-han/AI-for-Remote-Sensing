# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:24:50 2018

@author: Juhyun Lee
"""

# ! pip install numpy
# ! pip install scipy
# ! pip install matplotlib
# ! pip install scikit-learn

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm


digits = datasets.load_digits()
print(digits.data)
print(digits.target)

clf = svm.SVC(gamma=0.001, C=100)



####reset
#Import Library
import csv
import pandas as pd
import numpy as np
import scipy
from __future__ import print_function

from sklearn import datasets
#from sklearn.model_selection import train_test_split # trainig/testing 데이터셋으로 나눠있지 않을때 이 유틸으로 나눌수있음

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from time import time
from scipy.stats import randint as sp_randint
from scipy.stats.distributions import expon
from sklearn import svm, grid_search
from sklearn.svm import SVC
from io import StringIO





# create empty variables

# import training dataset
f_trn = pd.read_csv('climate_zone_cali.csv')
f_trn2 = np.array(f_trn)

f_trn.shape # number of rows, columns

X_trn_ins = f_trn2[:,0:-1]#,	T_sM,	T_sm,	T_wM,	T_wm,	P_ann, P_sM, P_sm, P_wM, P_wm, dem
Y_trn = f_trn.target
##X_trn_ins = np.array(X_trn_ins)#.astype('float64')
##Y_trn.astype('str')
# standardization of variables
X_trn = preprocessing.scale(X_trn_ins, axis=0)
# verification of standardization 
X_trn.mean(axis=0)

# import testing dataset
f_tst = pd.read_csv('climate_zone_vali.csv')
f_tst2 = np.array(f_tst)

f_tst.shape # number of rows, columns

X_tst_ins = f_tst2[:,0:-1]#,	T_sM,	T_sm,	T_wM,	T_wm,	P_ann, P_sM, P_sm, P_wM, P_wm, dem
Y_tst = f_tst.target
##X_tst_ins.astype('float64')
##Y_tst.astype('str')
# standardization of variables
X_tst = preprocessing.scale(X_tst_ins, axis=0)
# verification of standardization 
X_tst.mean(axis=0) # axis=0 -> mean of each column, axis=1 -> mean of each row



clf = svm.SVC() 
# kernel =  ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
# decision_function_shape = ‘ovo’, ‘ovr’, default=’ovr’

####### Parameter Optimization #######

# parameter ranking
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# exhaustive Grid Search
tuned_parameters = [{'kernel': ['poly'], 'degree': [1,2,3],'C': [1, 10, 100, 1000],'decision_function_shape' : ['ovr', 'ovo']},
                    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000],'decision_function_shape' : ['ovr', 'ovo']},
                    {'kernel': ['sigmoid'],'C': [1, 10, 100, 1000],'decision_function_shape' : ['ovr', 'ovo']},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000],'decision_function_shape' : ['ovr', 'ovo']}
                    ]

grid_search = GridSearchCV(clf, param_grid=tuned_parameters)
start = time()
grid_search.fit(X_trn, Y_trn)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)




# randomized parameter optimization
dist_parameters = {"kernel": ['poly', 'rbf', 'sigmoid', 'linear'],
                   "decision_function_shape": ['ovr', 'ovo'],
                   "C": scipy.stats.expon(scale=1000), #scipy.stats.expon : An exponential continuous random variable. 즉 0~1사이의 값을 랜덤하게 추출
                   "degree": scipy.stats.expon(scale=10),# scale = 1/lamda 즉 
                   "gamma": scipy.stats.expon(scale=.1)}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=dist_parameters,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_trn, Y_trn)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)


######### cf.Grid search 시행시 모든 모델들에 대한 score를 알고싶을때 #############
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, 
                       scoring='%s_macro' % score)
    clf.fit(X_trn, Y_trn)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    ## 각 모델에 의한 validation score 
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_tst, clf.predict(X_tst)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.



# 분류 모델 결과 보여주기
    
svm_model = SVC(kernel = 'linear', decision_function_shape = 'ovr', C = 1, degree = 3).fit(X_trn, Y_trn)
svm_predictions = svm_model.predict(X_tst)
 
# model accuracy for X_test  
accuracy = svm_model.score(X_tst, Y_tst)
 
# creating a confusion matrix
cm = confusion_matrix(Y_tst, svm_predictions)

