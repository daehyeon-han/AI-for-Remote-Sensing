import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os

## Read data
os.chdir('/Users/dhan/Dropbox/Archive/_coursework/2018_1st/AI_RS/week3/Week3_lab2/data')
cali = np.array(pd.read_csv('classification_tr.csv', dtype='float32'))
vali = np.array(pd.read_csv('classification_va.csv', dtype='float32'))

X_cali = cali[:,:-1]
Y_cali = cali[:,-1]
X_vali = vali[:,:-1]
Y_vali = vali[:,-1]

## Extra tree - Regression
etr = ExtraTreesRegressor(n_estimators=500, max_depth=None, min_samples_split=2,bootstrap=True)
etr.fit(X_cali,Y_cali)
yhat_vali = etr.predict(X_vali)
mean_squared_error(Y_vali, yhat_vali) # error

## Extra tree - Classification
etc = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=2,bootstrap=True)
etc.fit(X_cali,Y_cali)
yhat_vali = etc.predict(X_vali)
accuracy_score(Y_vali,yhat_vali,normalize=False)
confusion_matrix(Y_vali,yhat_vali)


## Feature importance
importances = etc.feature_importances_
std = np.std([tree.feature_importances_ for tree in etc.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X_cali.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest, 빨강 : feature importance, 검은선 : inter-trees variablity
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_cali.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(X_cali.shape[1]), indices)
plt.xlim([-1, X_cali.shape[1]])
plt.show()
