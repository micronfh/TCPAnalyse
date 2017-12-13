from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import GRU, LSTM
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import keras

TRAINING = "../../../../datas/LSTM/loss2_train_LSTM.csv"
TEST = "../../../../datas/LSTM/loss2_test_LSTM.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# 设置seed，重现结果
np.random.seed(2017)

learning_rate = 0.001
clip_norm = 1.0

# 总共有4个状态
nb_classes = 4

X_train = training_set.data
y_train = training_set.target
X_test = test_set.data
y_test = test_set.target

# 根据提取的特征，使用time为8，每个时间步长为16
X_train = X_train.reshape(X_train.shape[0], 8, -1)
X_test = X_test.reshape(X_test.shape[0], 8, -1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
in_shp = list(X_train.shape[1:])

print('x_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('x_test shape:', X_test.shape)

Y_train = np_utils.to_categorical(y_train - 1, nb_classes)
Y_test = np_utils.to_categorical(y_test - 1, nb_classes)

model = Sequential()
# model.add(Embedding(max_features, 128))
batch_size = 64
hidden_units = 32
dense_units = 32
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# GRU1 (GRU)                   (None, 10, 150)           71550
# _________________________________________________________________
# GRU2 (GRU)                   (None, 100)               75300
# _________________________________________________________________
# dense1 (Dense)               (None, 64)                6464
# _________________________________________________________________
# dense2 (Dense)               (None, 4)                 260
# _________________________________________________________________
# activation_1 (Activation)    (None, 4)                 0
# =================================================================

model.add(LSTM(hidden_units,
               input_shape=(X_train.shape[1], X_train.shape[2]),
               kernel_initializer='glorot_uniform',
               activation='relu',
               name='LSTM'
               ))

model.add(Dense(dense_units,
                kernel_initializer="he_normal",
                name='Dense'))

# model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation="softmax"))

# adam = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.summary()


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


# model.fit(X_train,
#           Y_train,
#           epochs=10,
#           batch_size=batch_size,
#           verbose=2)

nb_epoch = 100
patience = 10

filepath = 'E:\\GitHub\\PyCharmProjects\\TCPAnalyse\\models\\gru180.wts.h5'

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

# model.load_weights(filepath)
print("end train")

print("history.history['loss'] : ", history.history['loss'])
print("history.history['val_loss'] : ", history.history['val_loss'])

filepath = 'E:\\GitHub\\PyCharmProjects\\TCPAnalyse\\models\\lstm_dense.wts.h5'
model.save_weights(filepath)
model.load_weights(filepath)
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
