# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 22:26:07 2018

@author: cheolhee Yoo
"""
import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

tf.set_random_seed(777)  # for reproducibility

# For calibration
xy = np.loadtxt('/Users/cheolheeyoo/Downloads/Lab3/Logistic_Python/Heatwave_norm_train.csv', delimiter=',', dtype=np.float32,skiprows=1)
x_data =xy[:, 0:-1]
y_data =xy[:,[-1]]

# For test
xy_te = np.loadtxt('/Users/cheolheeyoo/Downloads/Lab3/Logistic_Python/Heatwave_norm_test.csv', delimiter=',', dtype=np.float32,skiprows=1)
x_data_te =xy_te[:, 0:-1]
y_data_te =xy_te[:,[-1]]


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 6])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([6, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10000):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(step, cost_val)

# Accuracy report
h, c, a = sess.run([hypothesis, predicted, accuracy],
                   feed_dict={X: x_data, Y: y_data})
#print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
print(a)

# Launch graph for test
h_te, c_te, a_te = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data_te, Y: y_data_te})
print(a_te)


CM = confusion_matrix(c_te,y_data_te)
np.set_printoptions(threshold=np.nan)

print("Confusion Matrix")
print(CM)
