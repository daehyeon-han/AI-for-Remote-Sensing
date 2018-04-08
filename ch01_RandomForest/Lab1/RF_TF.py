# class :   AI for Remote Sensing
# prof. :   Dr. Jungho Im (ersgis@unist.ac.kr)
# date  :   7, March, 2018
# TA    :   Daehyeon Han (dhan@unist.ac.kr)
# objectives:
#   1. To load Tensorflow and learn how to use it.
#   2. To run Random Forest with your own retely sensed data in Python.

# Import libraries
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest   # Random forest in TF
from tensorflow.python.ops import resources
import numpy as np
import pandas as pd

# Ignore all GPUs, tf random forest does not benefit from it.
# It is possible to select which GPU will be used, which is much faster in neural nets.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load wildfire data
work_path = '/Users/dhan/Dropbox/Archive/_coursework/2018_1st/AI_RS/week2/lab/Lab1'     # Define your work path
cali_path = work_path + '/' + 'cali.csv'
vali_path = work_path + '/' + 'vali.csv'
cali = np.array(pd.read_csv(cali_path, dtype='float32'))
vali = np.array(pd.read_csv(vali_path, dtype='float32'))

cali.shape      # You can check the shape of calibration dataset. [15707 samples, 19 variables, 1 label]
vali.shape      # You can check the shape of validataion dataset. [4266 samples, 19 variables, 1 label]

# Split your data into X and Y. Here, the last column is the true value.
X_cali = cali[:,:-1]
Y_cali = cali[:,-1]
X_vali = vali[:,:-1]
Y_vali = vali[:,-1]

# Parameters
num_steps = 100    # Total steps to train
num_classes = 2    # The binary wildfire detection
num_features = 19  # Total 19 variables
num_trees = 100
max_nodes = 1000

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])


# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# The the prediction.
infer_op= forest_graph.inference_graph(X)

# Compare prediction and true value
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    _, l = sess.run([train_op, loss_op], feed_dict={X: X_cali, Y: Y_cali})
    if i % 10 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: X_cali, Y: Y_cali})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test Model
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X_vali, Y: Y_vali}))    # vali accuracy
pred = sess.run(tf.argmax(infer_op,1), feed_dict={X: X_vali, Y: Y_vali})            # binary prediction results


Step 1, Loss: -0.000000, Acc: 0.886986
Step 10, Loss: -28.320000, Acc: 0.958551
Step 20, Loss: -217.600006, Acc: 0.980262
Step 30, Loss: -540.280029, Acc: 0.988985
Step 40, Loss: -928.460022, Acc: 0.992996
Step 50, Loss: -998.000000, Acc: 0.993506
Step 60, Loss: -998.000000, Acc: 0.993506
Step 70, Loss: -998.000000, Acc: 0.993506
Step 80, Loss: -998.000000, Acc: 0.993506
Step 90, Loss: -998.000000, Acc: 0.993506
Step 100, Loss: -998.000000, Acc: 0.993506

Validation Accuracy: 0.977726