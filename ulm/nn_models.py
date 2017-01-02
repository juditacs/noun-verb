from datetime import datetime
from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.optimizers import SGD, RMSprop

from experiment import Result


class FFNN:
    def __init__(self):
        pass


class SingleLayerRNN:

    def __init__(self, input_dim, output_dim, max_len, cells,
                 cell_type='LSTM',
                 activation='sigmoid', loss=None,
                 optimizer='rmsprop', lr=0.001,
                 nb_epoch=300, batch_size=64,
                 metrics=['accuracy']):
        model = Sequential()
        if cell_type == 'LSTM':
            model.add(LSTM(cells, input_shape=(max_len, input_dim)))
        elif cell_type == 'GRU':
            model.add(GRU(cells, input_shape=(max_len, input_dim)))
        model.add(Dense(output_dim, activation=activation))
        o = self.init_optimizer(optimizer, lr)
        model.compile(loss=loss, optimizer=o,
                      metrics=metrics)
        self.model = model
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

    def init_optimizer(self, name, lr):
        name_map = {
            'rmsprop': RMSprop,
            'SGD': SGD,
        }
        return name_map[name](lr)

    def train_and_test(self, X, y):
        result = Result()
        result.X_shape = X.shape
        result.y_shape = y.shape
        result.timestamp = datetime.now()
        start = datetime.now()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=.9)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=2)
        self.model.fit(X_train, y_train, callbacks=[early_stopping],
                       validation_split=.2,
                       nb_epoch=self.nb_epoch, verbose=0,
                       batch_size=self.batch_size)
        result.running_time = datetime.now() - start
        # Final evaluation of the model
        scores = self.model.evaluate(X_train, y_train, verbose=0)
        result.train_acc = scores[1]
        scores = self.model.evaluate(X_test, y_test, verbose=0)
        result.test_acc = scores[1]
        result.success = True
        return result
