#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin
from collections import defaultdict
import random


def parse_args():
    p = ArgumentParser()
    p.add_argument('mapping', type=str)
    p.add_argument('-r', '--random', action='store_true', default=False,
                   help='Randomize tags')
    p.add_argument('-m', '--max', dest='max_per_tag', type=int, default=0)
    return p.parse_args()


def read_mapping(fn):
    with open(fn) as f:
        mapping = {}
        for l in f:
            src, tgt = l.strip().split('\t')
            mapping[src] = tgt
        return mapping


def main():
    args = parse_args()
    mapping = read_mapping(args.mapping)
    tags = list(mapping.values())
    replaced = defaultdict(lambda: defaultdict(int))
    m = args.max_per_tag
    for l in stdin:
        fd = l.strip().split('\t')
        if len(fd) < 2:
            continue
        word, tag = fd[:2]
        tag = '/'.join(tag.split('/')[1:])
        if args.random:
            repl = random.choice(tags)
        else:
            repl = mapping.get(tag, 'NONE')
        if m > 0 and replaced[repl][tag] >= m:
            continue
        replaced[repl][tag] += 1
        print("{0}\t{1}".format(word, repl))

if __name__ == '__main__':
    main()
