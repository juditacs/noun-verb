#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
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
        'N': 2,
        'last_char': 6,
        'include_smaller_ngrams': True,
        'max_sample_per_class': 30000,
        'grep_filter': ("NOUN", "VERB"),
        'use_padding': True,
    },
    'model': {
        'type': 'ffnn',
        'layers': (20, 10),
        'activations': 'sigmoid',
        'optimizer': 'rmsprop',
        'optimizer_kwargs': {
            'lr': 0.001,
        },
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'nb_epoch': 300,
        'batch_size': 500,
        'early_stopping': True,
    },
}


def main():
    e = Experiment(ffnn_cfg)
    print(e.featurizer.X.shape)
    e.run_and_save()
    print(e.result)

if __name__ == '__main__':
    main()
