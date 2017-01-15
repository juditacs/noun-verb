#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
from sys import argv

from experiment import Experiment


cfg = {
    'global': {
        'nolog': False,
        'save_history': True,
        # 'comment': "Monitor val_acc for early stopping instead of val_loss",
    },
    'featurizer': {
        'type': 'character_sequence',
        'input_file': argv[1],
        'max_len': 11,
        'max_sample_per_class': 30000,
        'grep_filter': ("NOUN", "VERB"),
    },
    'model': {
        'type': 'rnn',
        'max_len': 11,
        'cell_type': 'GRU',
        'cell_num': 150,
        'optimizer': 'adam',
        'optimizer_kwargs': {
            'lr': .001,
        },
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'nb_epoch': 500,
        'batch_size': 256,
        'early_stopping': True,
        'early_stopping_patience': 10,
        'early_stopping_monitor': 'val_acc',
    },
}

cfg = {'featurizer': {'grep_filter': ('NOUN', 'VERB'), 'last_char': 10, 'N': 2, 'type': 'ngram', 'use_padding': True, 'include_smaller_ngrams': False, 'input_file': '/mnt/store/hlt/Language/Hungarian/Corp/webcorp_analed/xaa.tagged.utf8', 'max_sample_per_class': 200000},
       'model': {'early_stopping_monitor': 'val_acc', 'nb_epoch': 1, 'optimizer': 'adam', 'layers': (50, 30), 'loss': 'binary_crossentropy', 'early_stopping_patience': 2, 'optimizer_kwargs': {'lr': 0.001}, 'batch_size': 64, 'early_stopping': True, 'type': 'ffnn', 'activations': 'sigmoid', 'metrics': ['accuracy']}, 'global': {'df_path': 'results_cuda.tsv', 'save_history': True, 'nolog': True}}

cfg = {
    'featurizer': {
        'grep_filter': ('NOUN', 'VERB'),
        'include_smaller_ngrams': False,
        'type': 'ngram',
        'max_sample_per_class': 30000,
        'last_char': 6,
        'input_file': '/mnt/store/hlt/Language/Hungarian/Corp/webcorp_analed/xaa.tagged.utf8',
        'N': 3,
        'bagof': False,
        'use_padding': False},
    'model': {'activations': 'sigmoid', 'early_stopping_monitor': 'val_acc', 'optimizer_kwargs': {'lr': 1}, 'type': 'ffnn', 'batch_size': 512, 'layers': (10, 60), 'loss': 'binary_crossentropy', 'optimizer': 'rmsprop', 'early_stopping': True, 'early_stopping_patience': 2, 'metrics': ['accuracy'], 'nb_epoch': 300},
    'global': {'df_path': 'alma.tsv', 'nolog': True, 'save_history': True}}

def main():
    e = Experiment(cfg)
    print(e.featurizer.X.shape)
    e.run_and_save()
    print(e.result)

if __name__ == '__main__':
    main()
