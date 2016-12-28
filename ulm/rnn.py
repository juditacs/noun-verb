#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from datetime import datetime
import os

import pandas as pd

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.optimizers import SGD, RMSprop

from sklearn.model_selection import train_test_split

from featurize import CharacterSequenceFeaturizer


class SimpleLSTM:
    def __init__(self, input_dim, output_dim, max_len=10, cells=100,
                 activation='sigmoid', loss='binary_crossentropy',
                 optimizer='rmsprop',
                 lr=0.001,
                 nb_epoch=300, batch_size=64,
                 metrics=['accuracy'], **kwargs):
        model = Sequential()
        model.add(LSTM(cells, input_shape=(max_len, input_dim)))
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


class SimpleGRU(SimpleLSTM):
    def __init__(self, input_dim, output_dim, max_len=10, cells=100,
                 activation='sigmoid', loss='binary_crossentropy',
                 optimizer='rmsprop',
                 lr=0.001,
                 nb_epoch=300, batch_size=64,
                 metrics=['accuracy'], **kwargs):
        model = Sequential()
        model.add(GRU(cells, input_shape=(max_len, input_dim)))
        model.add(Dense(output_dim, activation=activation))
        o = self.init_optimizer(optimizer, lr)
        model.compile(loss=loss, optimizer=o,
                      metrics=metrics)
        self.model = model
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size


class Result:
    __slots__ = ('X_shape', 'y_shape',
                 'class_no',
                 'success', 'exception',
                 'nb_epoch', 'batch_size',
                 'running_time', 'timestamp',
                 'train_acc', 'test_acc')

    def to_dict(self):
        d = {}
        for k in Result.__slots__:
            try:
                d['result.{}'.format(k)] = getattr(self, k)
            except AttributeError:
                d['result.{}'.format(k)] = None
        return d

    def __str__(self):
        if self.success:
            return "SUCCESS: Train accuracy: {}\nTest accuracy: {}".format(
                self.train_acc, self.test_acc)
        else:
            return "FAIL: {}".format(self.exception)


class Experiment:

    def __init__(self, conf_d):
        self.conf_d = conf_d
        self.global_conf = conf_d['global']
        self.featurizer = CharacterSequenceFeaturizer(**conf_d['featurizer'])
        self.featurizer.featurize_file(self.global_conf['data_path'])
        idim, max_len, fdim, odim = self.featurizer.data_dim
        if self.conf_d['model']['rnn_type'] == 'lstm':
            self.model = SimpleLSTM(input_dim=fdim,
                                    output_dim=odim,
                                    max_len=max_len,
                                    **conf_d['model'])
        elif self.conf_d['model']['rnn_type'] == 'gru':
            self.model = SimpleGRU(input_dim=fdim, output_dim=odim,
                                   max_len=max_len,
                                   **conf_d['model'])
        else:
            raise ValueError("RNN type {} not supported".format(
                self.conf_d['model']['rnn_type']))

    def run_all(self):
        X, y = self.featurizer.create_matrices()
        result = self.model.train_and_test(X, y)
        if self.global_conf['nolog'] is False:
            self.save_experiment(result)
        return result

    def config_as_dict(self):
        d = {}
        d.update(self.serialize_section(
            self.conf_d['global'], 'global'))
        d.update(self.serialize_section(
            self.conf_d['model'], 'model'))
        d.update(self.serialize_section(
            self.conf_d['featurizer'], 'featurizer'))
        return d

    def serialize_section(self, section, pre):
        d = {}
        for k, v in section.items():
            d['{0}.{1}'.format(pre, k)] = v
        return d

    def save_experiment(self, result, fn='rnn_results.tsv'):
        data = self.config_as_dict()
        data.update(result.to_dict())
        if os.path.exists(fn):
            df = pd.read_table(fn)
        else:
            df = pd.DataFrame(columns=data.keys())
        new_cols = set(data.keys()) - set(df.columns)
        for c in new_cols:
            df[c] = None
        df = df.append(data, ignore_index=True)
        df = df.sort_index(axis=1)
        df.to_csv(fn, sep='\t', index=False)


def parse_args():
    p = ArgumentParser()
    p.add_argument('--max-len', type=int)
    p.add_argument('-s', '--sample-size', type=int)
    p.add_argument('-i', '--input-file', type=str)
    p.add_argument('-n', '--nb-epoch', type=int)
    p.add_argument('-b', '--batch-size', type=int)
    p.add_argument('--lstm-cells', type=int)
    return p.parse_args()


def main():
    cfg = {
        'global': {
            'data_path': "/mnt/store/hlt/Language/Hungarian/Corp/" +
            "webcorp_analed/xaa.tagged.utf8",
            'nolog': False,
        },
        'featurizer': {
            'max_len': 6,
            'max_sample_per_class': 30000,
            'label_filter': ("NOUN", "VERB"),
        },
        'model': {
            'lr': 0.1,
            'rnn_type': 'gru',
            'cells': 125,
            'batch_size': 512,
            'nb_epoch': 300,
        },

    }
    cfg['featurizer']['max_len'] = 9
    for nb in [500]:
        cfg['model']['nb_epoch'] = nb
        for lr in [0.01]:
            cfg['model']['lr'] = lr
            for cells in [1, 2, 5, 10, 20, 25, 50, 75, 100, 125, 150]:
                cfg['model']['cells'] = cells
                for typ in ['gru', 'lstm']:
                    cfg['model']['rnn_type'] = typ
                    e = Experiment(cfg)
                    print(typ, nb, lr, cells)
                    r = e.run_all()
                    print(r)


if __name__ == '__main__':
    main()
