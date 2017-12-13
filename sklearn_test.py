from __future__ import print_function

import keras.models as Models
import keras
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
from keras.layers import SimpleRNN, LSTM, GRU
from keras import initializers

from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import tree

TRAINING = "../datas/40ms_train.csv"
TEST = "../datas/40ms_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# learning_rate = 1e-6
clip_norm = 1.0

nb_classes = 4
# 设置seed，重现结果
np.random.seed(2017)

X_train = training_set.data
y_train = training_set.target
X_test = test_set.data
y_test = test_set.target
print(X_train.shape)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

from sklearn import metrics

y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
