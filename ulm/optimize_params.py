#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
import random
from sys import argv

from experiment import Experiment


ffnn_cfg = {
    'global': {
        'nolog': False,
        'save_history': True,
    },
    'featurizer': {
        'type': 'ngram',
        'input_file': argv[1],
        'N': None,
        'last_char': None,
        'max_sample_per_class': 30000,
        'grep_filter': ("NOUN", "VERB"),
        'include_smaller_ngrams': False,
    },
    'model': {
        'type': 'ffnn',
        'layers': None,
        'activations': 'sigmoid',
        'optimizer': 'rmsprop',
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'nb_epoch': None,
        'batch_size': None,
        'early_stopping': True,
    },
}
rnn_cfg = {
    'global': {
        'nolog': False,
        'save_history': True,
    },
    'featurizer': {
        'type': 'character_sequence',
        'input_file': argv[1],
        'max_len': None,
        'max_sample_per_class': 30000,
        'grep_filter': ("NOUN", "VERB"),
    },
    'model': {
        'type': 'rnn',
        'cell_type': None,
        'cell_num': None,
        'max_len': None,
        'optimizer': None,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'nb_epoch': None,
        'batch_size': None,
        'early_stopping': True,
    },
}
cnn_cfg = {
    'global': {
        'nolog': False,
        'save_history': True,
    },
    'featurizer': {
        'type': 'character_sequence',
        'input_file': argv[1],
        'max_len': None,
        'max_sample_per_class': 30000,
        'grep_filter': ("NOUN", "VERB"),
    },
    'model': {
        'type': 'cnn',
        'layers': None,
        'max_len': None,
        'optimizer': None,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'nb_epoch': None,
        'batch_size': None,
        'early_stopping': True,
    },
}
common_ranges = {
    'lr': (1, .1, .01, .001),
    'nb_epoch': (100, 300, 500, 1000),
    'batch_size': (256, 512, 1024, 2048),
    'optimizer': ('rmsprop') #, 'adam', 'sgd'),
}
ffnn_ranges = {
    'N': (1, 2, 3),
    'use_padding': (True, False),
    'last_char': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    'layers': [(i, j) for i in range(10, 61, 10)
                for j in range(10, 61, 10)],
}
rnn_ranges = {
    'max_len': (3, 4, 5, 6, 7, 8, 9, 10, 11),
    'cell_type': ('LSTM', 'GRU'),
    'cell_num': (1, 2, 5, 10, 20, 30, 40, 50, 75, 100,
                 125, 150),
}
cnn_ranges = {
    'max_len': (3, 4, 5, 6, 7, 8, 9, 10, 11),
    'nb_filter': (1, 2, 4, 8, 16, 32, 64, 128),
    'subsample_length': (1, 2, 3),
    'activation': ('linear', 'sigmoid', 'relu'),
    'filter_length': (1, 2, 3, 4, 5, 10, 20, 50),
}

def generate_common_params(d):
    d['model'].setdefault('optimizer_kwargs', {})
    d['model']['optimizer_kwargs']['lr'] = random.choice(common_ranges['lr'])
    d['model']['nb_epoch'] = random.choice(common_ranges['nb_epoch'])
    d['model']['batch_size'] = random.choice(common_ranges['batch_size'])
    d['model']['optimizer'] = random.choice(common_ranges['optimizer'])


def generate_ffnn_cfg():
    d = ffnn_cfg.copy()
    generate_common_params(d)
    d['featurizer']['N'] = random.choice(ffnn_ranges['N'])
    d['featurizer']['use_padding'] = random.choice(
        ffnn_ranges['use_padding'])
    d['featurizer']['last_char'] = random.choice(
        ffnn_ranges['last_char'])
    d['model']['layers'] = random.choice(ffnn_ranges['layers'])
    return d


def generate_rnn_cfg():
    d = rnn_cfg.copy()
    generate_common_params(d)
    ml = random.choice(rnn_ranges['max_len'])
    d['featurizer']['max_len'] = ml
    d['model']['max_len'] = ml
    d['model']['cell_type'] = random.choice(rnn_ranges['cell_type'])
    d['model']['cell_num'] = random.choice(rnn_ranges['cell_num'])
    return d


def generate_cnn_cfg():
    d = cnn_cfg.copy()
    generate_common_params(d)
    ml = random.choice(cnn_ranges['max_len'])
    d['featurizer']['max_len'] = ml
    d['model']['max_len'] = ml
    nb = random.choice(cnn_ranges['nb_filter'])
    f = random.choice(cnn_ranges['filter_length'])
    s = random.choice(cnn_ranges['subsample_length'])
    a = random.choice(cnn_ranges['activation'])
    d['model']['layers'] = [
        (nb, f, s, a),
        (None, None, None, a)
    ]
    return d

def main():
    exp_num = int(argv[2]) if len(argv) > 2 else 1
    print("Running FFNN experiments")
    for i in range(exp_num):
        if i+1 % 10 == 0:
            print("{}/{}".format(i+1, exp_num))
        d = generate_ffnn_cfg()
        e = Experiment(d)
        e.run_and_save()
    return
    print("Running RNN experiments")
    for _ in range(exp_num):
        d = generate_rnn_cfg()
        e = Experiment(d)
        e.run_and_save()
    print("Running CNN experiments")
    for _ in range(exp_num):
        d = generate_cnn_cfg()
        e = Experiment(d)
        e.run_and_save()

if __name__ == '__main__':
    main()
