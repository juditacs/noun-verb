#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from rnn import Experiment

general_filters = [
    ("<PLUR", ),
    ("<ACC", ),
]
verb_filters = [
    ("<PAST", ),
    ("<COND", ),
    ("<PERS<1", "<PERS<2", ""),
]
noun_filters = [
    ("<PLUR", ),
    ("<CAS<ACC", "<CAS<DAT", "<CAS<INE", "<CAS<INS", "<CAS<SBL"),
    ("<CAS<ACC", "<CAS<DAT", "<CAS<INE", "<CAS<INS", "<CAS<SBL", "<CAS<SUE", "<CAS<ALL",
     "<CAS<ILL", "<CAS<ELA", "<CAS<DEL"),
]
pos = [
    ("CONJ", "NOUN"),
    ("CONJ", "NOUN", "ADJ", "VERB"),
    ("ADV", "ADJ"),
]


def run_filter_with_params(cfg, pos_filt, grep_filt):
    for p in pos_filt:
        for g in grep_filt:
            cfg['featurizer']['label_filter'] = p
            cfg['featurizer']['grep_filter'] = g
            #cfg['global']['nolog'] = False
            if len(pos_filt) > 2 or len(grep_filt) > 2:
                cfg['model']['loss'] = 'categorical_crossentropy'
            else:
                cfg['model']['loss'] = 'binary_crossentropy'
            print(p, g)
            e = Experiment(cfg)
            print(e.run_all())


def main():
    cfg = {
        'global': {
            'data_path': ("/mnt/store/hlt/Language/Hungarian/Corp/"
                          "webcorp_analed/xaa.tagged.utf8"),
            'nolog': False,
        },
        'featurizer': {
            'max_len': 6,
            'max_sample_per_class': 10000,
            'label_filter': ("VERB", ),
            'grep_filter': ("<PAST", ""),
            'max_lines': 20000000,
        },
        'model': {
            'lr': 0.1,
            'rnn_type': 'gru',
            'cells': 5,
            'batch_size': 512,
            'nb_epoch': 300,
        },

    }
    for typ in ['gru', 'lstm']:
        cfg['model']['rnn_type'] = typ
        for ml in [1, 2, 3, 4, 5, 6, 7, 8]:
            cfg['featurizer']['max_len'] = ml
            for cells in [1, 2, 5, 10, 25, 50]:
                cfg['model']['cells'] = cells
                print(">>>>", typ, ml, cells)
                run_filter_with_params(cfg, pos, [None])
                run_filter_with_params(cfg, [None], general_filters)
                run_filter_with_params(cfg, [("VERB", )], verb_filters)
                run_filter_with_params(cfg, [("NOUN", )], noun_filters)


if __name__ == '__main__':
    main()
