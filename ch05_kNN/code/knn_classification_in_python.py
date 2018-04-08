# -*- coding: utf-8 -*-
"""
@author: seongmun sim
"""
#!pip install sklearn
#!pip install matplotlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
os.chdir('D:/CLASS/ML_DEEP_IMJ/Lab/Week6_lab5_kNN/code')

# Calib
cali = np.loadtxt('../data/classification_tr.csv', delimiter=',', dtype=np.float32,skiprows=1)
X_train =cali[:, 0:-1]
Y_train =cali[:,[-1]]

# Valid
vali = np.loadtxt('../data/classification_va.csv', delimiter=',', dtype=np.float32,skiprows=1)
X_test =vali[:, 0:-1]
Y_test =vali[:,[-1]]

# Set the k value
clf = KNeighborsClassifier(n_neighbors=1)
#n_neighbors=,
#weights=’uniform’, 'distance'
#algorithm=‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’,
#leaf_size=30, - for the computional optimize
#p=2, - how power?
#metric=’minkowski’,


# Fit the train set
clf.fit(X_train, Y_train)

# 결과 보기
print("테스트 세트 예측: {}".format(clf.predict(X_test)))
print("테스트 세트 정확도: {:.2f}".format(clf.score(X_test, Y_test)))

# 루프를 이용하여 k 최적화
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # 모델 생성
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, Y_train)
    # 훈련 세트 정확도 저장
    training_accuracy.append(clf.score(X_train, Y_train))
    # 일반화 정확도 저장
    test_accuracy.append(clf.score(X_test, Y_test))
    
# Tools → Preferences → Ipython Console → Graphics → Graphics Backend → Backend: “automatic”
plt.plot(neighbors_settings, training_accuracy, label="Training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="Test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()




