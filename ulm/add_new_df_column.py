#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import pandas as pd


def parse_args():
    p = ArgumentParser()
    p.add_argument('-i', '--infile', type=str, required=True)
    p.add_argument('-c', '--new-column', type=str, required=True)
    p.add_argument('-d', '--default-value', type=str, default=None)
    return p.parse_args()


def add_new_col_with_default(fn, colname, default):
    df = pd.read_table(fn)
    df[colname] = default
    df = df.sort_index(axis=1)
    df.to_csv(fn, sep='\t', index=False)


if __name__ == '__main__':
    args = parse_args()
    add_new_col_with_default(args.infile, args.new_column,
                             args.default_value)
