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

TRAINING = "../test/PC_100ms_loss1%_video1_feature40.csv"
TEST = "../test/PC_100ms_loss1%_video2_feature40.csv"

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

X_train = X_train.reshape(X_train.shape[0], 10, -1)
X_test = X_test.reshape(X_test.shape[0], 10, -1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
in_shp = list(X_train.shape[1:])

print('x_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('x_test shape:', X_test.shape)

Y_train = np_utils.to_categorical(y_train - 1, nb_classes)
Y_test = np_utils.to_categorical(y_test - 1, nb_classes)

print('buile model……')
dr = 0.5  # dropout rate
hidden_units1 = 128
hidden_units2 = 100

dense1_units = 64
learning_rate = 5 * 1e-4


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


model = Models.Sequential()

model.add(GRU(hidden_units1,
              activation='relu',
              input_shape=in_shp,
              return_sequences=True,
              name='GRU1'))
model.add(GRU(hidden_units2,
              kernel_initializer='glorot_uniform',
              activation='relu',
              name='GRU2'))
model.add(Dense(dense1_units, kernel_initializer="he_normal", name='dense1'))
model.add(Dense(nb_classes, kernel_initializer="he_normal", name='dense2'))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate, clipnorm=0.5)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
model.summary()

batch_size = 64
nb_epoch = 6
patience = 20

filepath = 'E:\\GitHub\\PyCharmProjects\\TCPAnalyse\\models\\gru180.wts.h5'

# history = model.fit(X_train,
#                     Y_train,
#                     batch_size=batch_size,
#                     epochs=nb_epoch,
#                     verbose=2,
#                     validation_data=[X_test, Y_test],
#                     callbacks=[
#                         keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
#                                                         mode='auto'),
#                         keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')
#                     ])

model.load_weights(filepath)
print("end train")

# print("history.history['loss'] : ", history.history['loss'])
# print("history.history['val_loss'] : ", history.history['val_loss'])

Y_test_hat = model.predict(X_test, batch_size=batch_size)
# model.reset_states()
conf = np.zeros([nb_classes, nb_classes])
confnorm = np.zeros([nb_classes, nb_classes])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    # k = list(Y_test_hat[i, :]).index(1)
    k = np.argmax(Y_test_hat[i, :])
    conf[j, k] = conf[j, k] + 1

for i in range(0, nb_classes):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

print("1: ", confnorm.shape, confnorm)
plt.figure()

ind_array = np.arange(nb_classes)
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = confnorm[y_val][x_val]
    if c >= 0.4:
        plt.text(x_val, y_val, "%0.3f" % (c,), color='white', fontsize=8, va='center', ha='center')
    elif c >= 0.01:
        plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=8, va='center', ha='center')
    else:
        plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=8, va='center', ha='center')

plot_confusion_matrix(confnorm, labels=[1, 2, 3, 4], title="ConvNet Confusion Matrix")
plt.show()
score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)

print('Test score:', score)
print('Test accuracy:', acc)
