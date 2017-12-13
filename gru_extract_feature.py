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

# TRAINING = "../test/PC_100ms_loss1%_video1_feature40.csv"
# TEST = "../test/PC_100ms_loss1%_video2_feature40.csv"

TRAINING = "../datas/feature/train_feature.csv"
TEST = "../datas/feature/test_feature.csv"

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
hidden_units1 = 150
hidden_units2 = 100

dense1_units = 64
learning_rate = 5 * 1e-4
batch_size = 64
nb_epoch = 1000
patience = 5


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
              kernel_initializer='glorot_uniform',
              activation='relu',
              input_shape=in_shp,
              return_sequences=True,
              name='GRU1'))

model.add(GRU(hidden_units2,
              kernel_initializer='glorot_uniform',
              activation='relu',
              name='GRU2'))
model.add(Dense(dense1_units,
                kernel_initializer="he_normal",
                name='dense1'))
model.add(Dense(nb_classes,
                kernel_initializer="he_normal",
                name='dense2'))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate, clipnorm=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])
model.summary()

# filepath = 'E:\\GitHub\\PyCharmProjects\\TCPAnalyse\\models\\gru_feature.wts.h5'
filepath = 'E:\\GitHub\\PyCharmProjects\\TCPAnalyse\\models\\gru_multi.wts.h5'

history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=2,
                    validation_data=[X_test, Y_test],
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                                        mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=0, mode='auto')
                    ])
print("history.history['loss'] : ", history.history['loss'])
print("history.history['val_loss'] : ", history.history['val_loss'])

model.load_weights(filepath)
print("end train")
#
# score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
# print("score:", score)

# create model2--get output among layers !!!!
print("model2")
model2 = Models.Sequential()
model2.add(GRU(hidden_units1,
               kernel_initializer='glorot_uniform',
               # recurrent_initializer=initializers.Identity(gain=1.0),
               activation='relu',
               input_shape=in_shp,
               return_sequences=True,  # add_0425
               weights=model.layers[0].get_weights(),
               name='GRU2_1'
               ))
model2.add(GRU(hidden_units2,
               kernel_initializer='glorot_uniform',
               # recurrent_initializer=initializers.Identity(gain=1.0),
               activation='relu',
               # return_sequences=True,  # add_0425
               weights=model.layers[1].get_weights(),
               name='GRU2_2'
               ))
model2.add(Dense(dense1_units,
                 kernel_initializer='he_normal',
                 weights=model.layers[2].get_weights(),
                 name="dense2_1"))
# model2.add(Dense(len(classes), init='he_normal', weights=model.layers[3].get_weights(), name="dense2_2"))
# model2.add(Activation('softmax'))
# model2.add(Reshape([len(classes)]))

# 指定损失函数和优化函数
rmsprop = RMSprop(lr=learning_rate, clipnorm=0.5)  # 理论上adam优于RMSprop
model2.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
model2.summary()

print("begin save csv")  # save as weka
# train
feature_train = model2.predict(X_train)
label_train = np.zeros((feature_train.shape[0], 1), dtype=np.float32)
for i in range(0, label_train.shape[0]):
    j = list(Y_train[i, :]).index(1)
    label_train[i, 0] = j + 1
out_train = np.column_stack((feature_train, label_train))
np.savetxt("train.csv", out_train, fmt='%f', delimiter=',')

# test
feature_test = model2.predict(X_test)
label_test = np.zeros((feature_test.shape[0], 1), dtype=np.float32)
for i in range(0, label_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    label_test[i, 0] = j + 1
out_test = np.column_stack((feature_test, label_test))
np.savetxt("test.csv", out_test, fmt='%f', delimiter=',')
