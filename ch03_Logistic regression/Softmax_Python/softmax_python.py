#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:02:33 2018

@author: cheolheeyoo
"""

import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
tf.set_random_seed(777)  # for reproducibility

# For calibration
xy = np.loadtxt('/Users/cheolheeyoo/Downloads/Lab3/Softmax_Python/Warning_norm_train.csv', delimiter=',', dtype=np.float32 ,skiprows=1)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


# For test
xy_te = np.loadtxt('/Users/cheolheeyoo/Downloads/Lab3/Softmax_Python/Warning_norm_test.csv', delimiter=',', dtype=np.float32 ,skiprows=1)
x_data_te = xy_te[:, 0:-1]
y_data_te = xy_te[:, [-1]]


print(x_data.shape, y_data.shape)

nb_classes = 3  # 0 ~ 2

X = tf.placeholder(tf.float32, [None, 6])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 2
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([6, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Launch graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(30000):
    sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        loss, acc = sess.run([cost, accuracy], feed_dict={
                             X: x_data, Y: y_data})
        print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
            step, loss, acc))

# Let's see if we can predict
pred = sess.run(prediction, feed_dict={X: x_data})
# y_data: (N,1) = flatten => (N, ) matches pred.shape
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


# Launch graph for test
pred_te = sess.run(prediction, feed_dict={X: x_data_te})
for p, y in zip(pred_te, y_data_te.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

acc_te = sess.run(accuracy, feed_dict={X: x_data_te, Y:y_data_te})
print(acc_te)

CM = confusion_matrix(pred_te,y_data_te)
np.set_printoptions(threshold=np.nan)

print("Confusion Matrix")
print(CM)


