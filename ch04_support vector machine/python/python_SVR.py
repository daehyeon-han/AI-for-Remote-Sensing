# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:24:50 2018

@author: Juhyun Lee
"""

# ! pip install numpy
# ! pip install scipy
# ! pip install matplotlib
# ! pip install scikit-learn

#Import Library
import csv
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

from __future__ import print_function
from __future__ import division

from sklearn import datasets
#from sklearn.model_selection import train_test_split # trainig/testing 데이터셋으로 나눠있지 않을때 이 유틸으로 나눌수있음
from sklearn.svm import SVR
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
f_trn = pd.read_csv('PM10_cal_AIRS.csv')
f_trn2 = np.array(f_trn)

f_trn.shape # number of rows, columns

X_trn_ins = f_trn2[:,0:-1]
Y_trn = f_trn.PM10
##X_trn_ins = np.array(X_trn_ins)#.astype('float64')
##Y_trn.astype('str')
# standardization of variables
X_trn = preprocessing.scale(X_trn_ins, axis=0)
# verification of standardization 
X_trn.mean(axis=0)

# import testing dataset
f_tst = pd.read_csv('PM10_val_AIRS.csv')
f_tst2 = np.array(f_tst)

f_tst.shape # number of rows, columns

X_tst_ins = f_tst2[:,0:-1]
Y_tst = f_tst.PM10
##X_tst_ins.astype('float64')
##Y_tst.astype('str')
# standardization of variables
X_tst = preprocessing.scale(X_tst_ins, axis=0)
# verification of standardization 
X_tst.mean(axis=0) # axis=0 -> mean of each column, axis=1 -> mean of each row



clf = SVR() 
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
tuned_parameters = [{'kernel': ['poly'], 'degree': [1,2,3],'C': [1, 10, 100, 1000]}, # poly만 활성화 시켜 돌렸을때 1704초 걸림
                    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['sigmoid'],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                    ]

grid_search = GridSearchCV(clf, param_grid=tuned_parameters)
grid_search.fit(X_trn, Y_trn)


print("GridSearchCV took")
report(grid_search.cv_results_)


# randomized parameter optimization
dist_parameters = {"kernel": ['poly', 'rbf', 'sigmoid', 'linear'],
                   "C": scipy.stats.expon(scale=1000), #scipy.stats.expon : An exponential continuous random variable. 즉 0~1사이의 값을 랜덤하게 추출
                   "degree": scipy.stats.expon(scale=10),# scale = 1/lamda 즉 
                   "gamma": scipy.stats.expon(scale=.1)}

n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=dist_parameters,
                                   n_iter=n_iter_search)


random_search.fit(X_trn, Y_trn)
print("RandomizedSearchCV")
report(random_search.cv_results_)


# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1016.6346308825121, gamma=0.67967440780471999)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X_trn, Y_trn).predict(X_tst)
y_lin = svr_lin.fit(X_trn, Y_trn).predict(X_tst)
y_poly = svr_poly.fit(X_trn, Y_trn).predict(X_tst)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(Y_tst, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(Y_tst, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(Y_tst, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()