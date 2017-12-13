from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import keras

TRAINING = "../../../../datas/LSTM/50ms_train_LSTM.csv"
TEST = "../../../../datas/LSTM/50ms_test_LSTM.csv"

# 设置seed，重现结果
np.random.seed(2017)

learning_rate = 1e-6
clip_norm = 1.0

nb_classes = 4

# model.reset_states()
conf = np.zeros([nb_classes, nb_classes])
conf = np.array([(12052, 2230, 0, 0),
                 (2293, 1785729, 0, 0),
                 (0, 0, 2100, 55),
                 (0, 0, 2, 26)])

confnorm = np.zeros([nb_classes, nb_classes])

for i in range(0, nb_classes):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


print("1: ", confnorm.shape, confnorm)

plt.figure()
ind_array = np.arange(nb_classes)
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = confnorm[y_val][x_val]
    if c >= 0.4:
        plt.text(x_val, y_val, "%0.4f" % (c,), color='white', fontsize=8, va='center', ha='center')
    elif c >= 0.01:
        plt.text(x_val, y_val, "%0.4f" % (c,), color='black', fontsize=8, va='center', ha='center')
    else:
        plt.text(x_val, y_val, "%0.4f" % (c,), color='black', fontsize=8, va='center', ha='center')

plot_confusion_matrix(confnorm, labels=['SS phase', 'CA phase', 'FR phase', 'TR phase'],
                      title="Raw-DTC Confusion Matrix(Delay=200ms)")
# plot_confusion_matrix(confnorm, labels=['SS phase', 'CA phase', 'FR phase', 'TR phase'],
#                       title="ConvNet Confusion Matrix(Loss=1%)")
plt.show()
