#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
from sys import argv

from experiment import Experiment

webcorp_fn = '/mnt/store/hlt/Language/Hungarian/Corp/webcorp_analed/xaa.tagged.utf8'
noun_fn = '/mnt/store/judit/projects/ulm/vitmav45-2016-MorphoDeep/dat/webcorp_filtered/nouns'
verb_fn = '/mnt/store/judit/projects/ulm/vitmav45-2016-MorphoDeep/dat/webcorp_filtered/verbs'
noun_verb_fn = '/mnt/store/judit/projects/ulm/vitmav45-2016-MorphoDeep/dat/webcorp_filtered/nouns_and_verbs'


def set_verb_person_filter(d):
    d['global']['exp_type'] = 'verb_person'
    d['featurizer']['input_file'] = verb_fn 
    d['featurizer']['regex_filter'] = (r'<PERS<([12])>', )
    d['featurizer']['include_none_labels'] = True
    return d['global']['exp_type']

def set_singular_plural_filter(d):
    d['global']['exp_type'] = 'plural'
    d['featurizer']['input_file'] = noun_verb_fn
    d['featurizer']['regex_filter'] = (r'(<PLUR)', )
    d['featurizer']['include_none_labels'] = True
    return d['global']['exp_type']

def set_accusative_filter(d):
    d['global']['exp_type'] = 'accusative'
    d['featurizer']['input_file'] = noun_fn
    d['featurizer']['regex_filter'] = (r'(<ACC)', )
    d['featurizer']['include_none_labels'] = True
    return d['global']['exp_type']

def set_verbtense_filter(d):
    d['global']['exp_type'] = 'tense'
    d['featurizer']['input_file'] = verb_fn
    d['featurizer']['regex_filter'] = (r'(<PAST)', )
    d['featurizer']['include_none_labels'] = True
    return d['global']['exp_type']

def set_conditional_filter(d):
    d['global']['exp_type'] = 'conditional'
    d['featurizer']['input_file'] = verb_fn
    d['featurizer']['regex_filter'] = (r'(<COND)', )
    d['featurizer']['include_none_labels'] = True
    return d['global']['exp_type']

def set_noun_cases_filter(d):
    d['global']['exp_type'] = 'noun_cases10'
    d['featurizer']['input_file'] = webcorp_fn
    d['featurizer']['grep_filter'] = (
        "<CAS<ACC", "<CAS<DAT", "<CAS<INE", "<CAS<INS",
        "<CAS<SBL", "<CAS<SUE", "<CAS<ALL", "<CAS<ILL",
        "<CAS<ELA", "<CAS<DEL")
    return d['global']['exp_type']

def set_pos_filter(d):
    d['global']['exp_type'] = 'pos4'
    d['featurizer']['input_file'] = webcorp_fn
    d['featurizer']['grep_filter'] = (
        "NOUN", "ADJ", "ADV", "VERB",
    )
    return d['global']['exp_type']


gen_funcs = [
    set_verb_person_filter,
    set_singular_plural_filter,
    set_accusative_filter,
    set_verbtense_filter,
    set_conditional_filter,
    set_noun_cases_filter,
    set_pos_filter,
]


def run(func):
    cfg = {
        'global': {
            'nolog': False,
            'save_history': True,
        },
        'featurizer': {
            'type': 'character_sequence',
            'input_file': None,
            'max_sample_per_class': 10000,
            'max_len': 2,
            # 'grep_filter': ("NOUN", "VERB"),
        },
        'model': {
            'type': 'ffnn',
            'layers': (10, 10),
            'activations': 'sigmoid',
            'optimizer': 'rmsprop',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'nb_epoch': None,
            'batch_size': None,
            'early_stopping': True,
        },
    }
    func(cfg)
    e = Experiment(cfg)
    e.featurizer.dataset.write_to_file('../dat/webcorp_filtered/{}'.format(
        cfg['global']['exp_type']))

if __name__ == '__main__':
    for f in gen_funcs:
        print(f.__name__)
        run(f)
