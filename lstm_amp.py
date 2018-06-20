from __future__ import print_function

from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import scipy.io as sio
import time


def transformdata(x):
    resultData = []
    for idx in range(0, len(x)):
        tmp = x[idx]
        for row in tmp:
            resultData.append(row.tolist())
    resultData = np.asarray(resultData)
    return resultData


if __name__ == '__main__':
    maxlen = 800  # cut texts after this number of words
    batch_size = 64
    print('Loading data...')
    x_test_cell = sio.loadmat('./x_test.mat')
    x_train_cell = sio.loadmat('./x_train.mat')
    y_test_cell = sio.loadmat('./y_test.mat')
    y_train_cell = sio.loadmat('./y_train.mat')
    x_test_streams = x_test_cell['testDataSet']
    x_train_streams = x_train_cell['trainDataSet']
    y_test = y_test_cell['testLabel']
    y_train_all = y_train_cell['trainLabelAll']
    x_test_single = x_test_streams[:, 5]
    x_train_single = x_train_streams[:, 5]

    x_test = transformdata(x_test_single)
    x_train = transformdata(x_train_single)
    y_test = y_test[0]
    y_train = y_train_all[0]
    y_train_categorical = to_categorical(y_train, num_classes=7)
    y_train_categorical = y_train_categorical[:, 1:7]
    print(y_train_categorical)
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')
    x_train = x_train[:, :, np.newaxis]
    x_test = x_test[:, :, np.newaxis]
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, return_sequences=True,
                   input_shape=(800, 1)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(6, activation='softmax'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('Train...')
    model.fit(x_train, y_train_categorical,
              batch_size=batch_size,
              epochs=5)
    # model summary
    model.summary()
    start_time = time.time()
    y_predict_class = model.predict(x_test, batch_size=batch_size, verbose=1)
    print("--- %s seconds ---" % (time.time() - start_time))
    np.savetxt('y_predict_belllabs_LSTM.out', y_predict_class)
