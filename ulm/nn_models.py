from datetime import datetime

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Convolution1D, GlobalMaxPooling1D

from result import Result
from utils import create_list_if_str, densify


class NN_Model:
    def __init__(self, input_dim, output_dim,
                 optimizer, optimizer_kwargs, loss, metrics, nb_epoch,
                 lr, batch_size, early_stopping):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.early_stopping = early_stopping
        self.model_fit_args = {
            'nb_epoch': nb_epoch,
            'batch_size': batch_size,
        }
        self.model_compile_args = {
            'optimizer': optimizer,
            'metrics': metrics,
            'loss': loss,
        }
        self.create_network()
        self.model.compile(**self.model_compile_args)
        self.result = Result()

    def fit(self, X, y):
        X = densify(X)
        y = densify(y)
        start = datetime.now()
        if self.early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', patience=2)
            self.result.history = self.model.fit(X, y, verbose=0,
                                                 validation_split=0.2,
                                                 callbacks=[early_stopping],
                                                 **self.model_fit_args)
        else:
            self.result.history = self.model.fit(X, y, verbose=0,
                                                 validation_split=0.2,
                                                 **self.model_fit_args)
        self.result.running_time = datetime.now() - start
        return self.result

    def evaluate(self, X, y, prefix):
        r = self.model.evaluate(densify(X), densify(y), batch_size=128)
        setattr(self.result, '{}_loss'.format(prefix), r[0])
        setattr(self.result, '{}_acc'.format(prefix), r[1])

    def to_json(self):
        return self.model.to_json()


class FFNN(NN_Model):
    def __init__(self, input_dim, output_dim, layers,
                 activations='sigmoid', optimizer='rmsprop',
                 optimizer_kwargs={}, loss='binary_crossentropy',
                 metrics=['accuracy'], nb_epoch=300, lr=.01,
                 batch_size=128, early_stopping=True):
        self.layers = layers
        self.activations = create_list_if_str(activations,
                                              len(layers)+1)
        super().__init__(input_dim, output_dim,
                         optimizer=optimizer,
                         optimizer_kwargs=optimizer_kwargs,
                         loss=loss, metrics=metrics, lr=lr,
                         nb_epoch=nb_epoch, batch_size=batch_size,
                         early_stopping=early_stopping)

    def create_network(self):
            self.model = Sequential()
            # input layer
            self.model.add(Dense(self.layers[0], input_dim=self.input_dim,
                                 activation=self.activations[0]))
            for i in range(1, len(self.layers)):
                self.model.add(Dense(self.layers[i],
                                     activation=self.activations[i]))
            # output layer
            self.model.add(Dense(self.output_dim,
                                 activation=self.activations[-1]))


class SingleLayerRNN(NN_Model):

    def __init__(self, input_dim, output_dim,
                 cell_type, cell_num, max_len,
                 output_activation='sigmoid',
                 optimizer='rmsprop',
                 optimizer_kwargs={}, loss='binary_crossentropy',
                 metrics=['accuracy'], nb_epoch=300, lr=.01,
                 batch_size=128, early_stopping=True):
        self.cell_type = cell_type
        self.max_len = max_len
        self.cell_num = cell_num
        self.output_activation = output_activation
        super().__init__(input_dim, output_dim,
                         optimizer=optimizer,
                         optimizer_kwargs=optimizer_kwargs,
                         loss=loss, metrics=metrics, lr=lr,
                         nb_epoch=nb_epoch, batch_size=batch_size,
                         early_stopping=early_stopping)

    def create_network(self):
        model = Sequential()
        if self.cell_type == 'LSTM':
            model.add(LSTM(self.cell_num,
                           input_shape=(self.max_len, self.input_dim)))
        elif self.cell_type == 'GRU':
            model.add(GRU(self.cell_num,
                          input_shape=(self.max_len, self.input_dim)))
        model.add(Dense(self.output_dim, activation=self.output_activation))
        self.model = model


class Conv1D(NN_Model):
    def __init__(self, input_dim, output_dim, layers, max_len,
                 init='glorot_uniform', optimizer='rmsprop',
                 optimizer_kwargs={}, loss='binary_crossentropy',
                 metrics=['accuracy'], nb_epoch=300, lr=.01,
                 batch_size=128, early_stopping=True):
        self.layers = layers
        self.max_len = max_len
        super().__init__(input_dim, output_dim,
                         optimizer=optimizer,
                         optimizer_kwargs=optimizer_kwargs,
                         loss=loss, metrics=metrics, lr=lr,
                         nb_epoch=nb_epoch, batch_size=batch_size,
                         early_stopping=early_stopping)

    def create_network(self):
        model = Sequential()
        # input layer
        model.add(Convolution1D(self.layers[0][0], self.layers[0][1],
                                activation=self.layers[0][2],
                                input_shape=(self.max_len, self.input_dim)))
        for i in range(1, len(self.layers)-1):
            model.add(Convolution1D(self.layers[i][0], self.layers[i][1],
                                    activation=self.layers[i][2]))
        model.add(GlobalMaxPooling1D())
        # output layer
        model.add(Dense(self.output_dim,
                        activation=self.layers[-1][2]))
        self.model = model
