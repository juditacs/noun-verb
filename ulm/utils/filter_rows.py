#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>

from sys import argv, stdin


def read_table(filters):
    head = next(stdin)
    index_filter = {}
    d = {}
    for i, col in enumerate(head.strip().split('\t')):
        for filt in filters:
            if filt in col:
                index_filter[i] = col
                d[col] = []
    for line in stdin:
        for i, col in enumerate(line.strip().split('\t')):
            if i in index_filter:
                d[index_filter[i]].append(col)
    return d


def main():
    columns = argv[1:]
    data = read_table(columns)
    data_len = len(list(data.values())[0])
    print('\t'.join(data.keys()))
    for i in range(data_len):
        print('\t'.join(data[k][i] for k in sorted(data.keys())))

if __name__ == '__main__':
    main()
