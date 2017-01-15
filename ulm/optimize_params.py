#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
import random
from argparse import ArgumentParser

from experiment import Experiment
from run_experiment import gen_funcs


sample_per_class = 10000

common_ranges = {
    #'lr': (1, .1, .01, .001),
    'lr': (.01, ),
    'nb_epoch': (300, ),
    'batch_size': (512, ),
    'optimizer': ('rmsprop', )  # , 'adam'),
}
ffnn_ranges = {
    'N': (1, 2),
    'use_padding': (True, False),
    'last_char': (1, 2, 3, 4, 5, 6, 7, 8),
    'layers': [(i, j) for i in range(10, 61, 10)
               for j in range(10, 61, 10)],
}
rnn_ranges = {
    'max_len': (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    'cell_type': ('LSTM', 'GRU'),
    'cell_num': (1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150,
                 175, 200, 225, 250, 300, 400, 500),
}
cnn_ranges = {
    'max_len': (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    'nb_filter': (1, 2, 4, 8, 16, 32, 64, 128),
    'subsample_length': (1, 2, 3),
    'activation': ('linear', 'sigmoid', 'relu'),
    'filter_length': (1, 2, 3, 4, 5, 10, 20, 50),
}


def generate_common_params(d):
    d['global']['df_path'] = 'results_cuda.tsv'
    d['model'].setdefault('optimizer_kwargs', {})
    d['model']['optimizer_kwargs']['lr'] = random.choice(common_ranges['lr'])
    d['model']['nb_epoch'] = random.choice(common_ranges['nb_epoch'])
    d['model']['batch_size'] = random.choice(common_ranges['batch_size'])
    d['model']['optimizer'] = random.choice(common_ranges['optimizer'])
    d['model']['early_stopping'] = True
    d['model']['early_stopping_patience'] = 2
    d['model']['early_stopping_monitor'] = 'val_acc'
    d['featurizer']['grep_filter'] = None
    d['featurizer']['regex_filter'] = None
    d['global']['nolog'] = False


def generate_ffnn_cfg():
    ffnn_cfg = {
        'global': {
            'nolog': False,
            'save_history': True,
        },
        'featurizer': {
            'type': 'ngram',
            'input_file': args.input,
            'N': None,
            'last_char': None,
            'max_sample_per_class': sample_per_class,
            # 'grep_filter': ("NOUN", "VERB"),
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
    d = ffnn_cfg
    generate_common_params(d)
    d['featurizer']['N'] = random.choice(ffnn_ranges['N'])
    d['featurizer']['include_smaller_ngrams'] = random.choice((True, False))
    d['featurizer']['use_padding'] = random.choice(
        ffnn_ranges['use_padding'])
    if d['featurizer']['use_padding'] is True:
        start = 1
    else:
        start = d['featurizer']['N']
    d['featurizer']['last_char'] = random.choice(
        range(start, max(ffnn_ranges['last_char'])+1))
    d['featurizer']['bagof'] = random.choice((True, False))
    d['model']['layers'] = random.choice(ffnn_ranges['layers'])
    return d


def generate_rnn_cfg():
    rnn_cfg = {
        'global': {
            'nolog': False,
            'save_history': True,
        },
        'featurizer': {
            'type': 'character_sequence',
            'input_file': args.input,
            'max_len': None,
            'max_sample_per_class': sample_per_class,
            # 'grep_filter': ("NOUN", "VERB"),
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
    d = rnn_cfg
    generate_common_params(d)
    ml = random.choice(rnn_ranges['max_len'])
    d['featurizer']['max_len'] = ml
    d['model']['max_len'] = ml
    d['model']['cell_type'] = random.choice(rnn_ranges['cell_type'])
    d['model']['cell_num'] = random.choice(rnn_ranges['cell_num'])
    return d


def generate_cnn_cfg():
    cnn_cfg = {
        'global': {
            'nolog': False,
            'save_history': True,
        },
        'featurizer': {
            'type': 'character_sequence',
            'input_file': args.input,
            'max_len': None,
            'max_sample_per_class': sample_per_class,
            # 'grep_filter': ("NOUN", "VERB"),
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
    d = cnn_cfg
    generate_common_params(d)
    ml = random.choice(cnn_ranges['max_len'])
    d['featurizer']['max_len'] = ml
    d['model']['max_len'] = ml
    nb = random.choice(cnn_ranges['nb_filter'])
    f = random.choice(range(1, ml+1))
    s = random.choice(cnn_ranges['subsample_length'])
    a = random.choice(cnn_ranges['activation'])
    d['model']['layers'] = [
        (nb, f, s, a),
        (None, None, None, a)
    ]
    return d


types = [
    'verb_person',
    'plural',
    'accusative',
    'tense',
    'conditional',
    'noun_cases10',
    'pos4',
]

def run_rnn():
    d = generate_rnn_cfg()
    g = random.choice(types)
    d['global']['exp_type'] = g
    d['featurizer']['input_file'] = '../dat/webcorp_filtered/{}'.format(g)
    print(d)
    e = Experiment(d)
    e.run_and_save()
    print('')
    print(e.result)


def run_ffnn():
    d = generate_ffnn_cfg()
    g = random.choice(types)
    d['global']['exp_type'] = g
    d['featurizer']['input_file'] = '../dat/webcorp_filtered/{}'.format(g)
    print(d)
    e = Experiment(d)
    e.run_and_save()
    print('')
    print(e.result)


def run_cnn():
    d = generate_cnn_cfg()
    g = random.choice(types)
    d['global']['exp_type'] = g
    d['featurizer']['input_file'] = '../dat/webcorp_filtered/{}'.format(g)
    print(d)
    e = Experiment(d)
    e.run_and_save()
    print('')
    print(e.result)


def parse_args():
    p = ArgumentParser()
    p.add_argument('-e', '--exp-num', default=1, type=int)
    p.add_argument('-i', '--input', type=str)
    p.add_argument('-t', '--types', type=str, default='FRC')
    return p.parse_args()


args = parse_args()



def main():
    exp_num = args.exp_num
    types = args.types
    for i in range(1, exp_num+1):
        print("EXPERIMENT {}/{}".format(i, exp_num))
        if 'F' in types:
            run_ffnn()
        if 'R' in types:
            run_rnn()
        if 'C' in types:
            run_cnn()

if __name__ == '__main__':
    main()
