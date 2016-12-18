#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import logging

from argparse import ArgumentParser
from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
import numpy as np


class InvalidTag(Exception):
    pass


class GrepFilter:
    def __init__(self, grep_filter, include_default=False):
        self.grep_filter = grep_filter
        if grep_filter is None:
            self.all_pass = True
        else:
            self.all_pass = False
            self.include_default = (len(self.grep_filter) == 1 or include_default)

    def match(self, label):
        if self.all_pass is True:
            return ""
        for gf in self.grep_filter:
            if gf in label:
                return gf
        if self.include_default:
            return 'default'
        return None
        raise RuntimeError("Label should have been skipped earlier: {}".format(
            label))

    def skip(self, label):
        # FIXME
        if label is None:
            return False
        if self.all_pass is True or self.include_default is True:
            return False
        if all(gf not in label for gf in self.grep_filter) and \
                self.include_default is False:
            return True
        return False


class Featurizer:
    def __init__(self, max_sample_per_class,
                 max_lines=0, skip_duplicates=True,
                 label_filter=None, grep_filter=None):
        self.max_sample_per_class = max_sample_per_class
        self.maximize_sample_per_class = max_sample_per_class > 0
        self.max_lines = max_lines
        self.line_cnt = 0
        self.skip_duplicates = skip_duplicates
        self.uniq_samples = set()
        self.samples = []
        self.labels = []
        self.sample_per_class = defaultdict(int)
        self.label_filter = None if label_filter is None else \
            set(label_filter)
        self.grep_filter = GrepFilter(grep_filter)
        self.raw_samples = []

    def featurize_file(self, fn):
        with open(fn) as f:
            for line in f:
                self.line_cnt += 1
                if self.continue_reading() is False:
                    break
                if self.skip_line(line):
                    continue
                try:
                    sample = self.extract_sample_from_line(line)
                except InvalidTag:
                    continue
                if self.skip_sample(sample):
                    continue
                self.featurize_and_store_sample(sample)
                self.update_counters(sample)

    def update_counters(self, sample):
        x, y = sample
        self.sample_per_class[y] += 1
        if self.skip_duplicates:
            self.uniq_samples.add((x, y))

    def continue_reading(self):
        if self.max_lines > 0 and self.line_cnt >= self.max_lines:
            return False
        if self.maximize_sample_per_class and len(self.sample_per_class) > 1:
            if all(v >= self.max_sample_per_class
                   for v in self.sample_per_class.values()):
                return False
        return True

    def skip_line(self, line):
        if self.skip_duplicates and line in self.uniq_samples:
            return True
        if not line.strip():
            return True
        if 'UNKNOWN' in line or '??' in line:
            return True
        return False

    def skip_sample(self, sample):
        word, (pos, tag) = sample
        label = (pos, tag)
        if self.grep_filter.skip(tag):
            return True
        if self.label_filter is not None and pos not in self.label_filter:
            return True
        if self.maximize_sample_per_class and \
                self.sample_per_class[label] >= self.max_sample_per_class:
            return True
        return False

    def extract_sample_from_line(self, line):
        fd = line.strip().split('\t')
        if len(fd) < 2:
            raise InvalidTag("Not enough fields.")
        word, tag = fd[:2]
        try:
            pos = tag.split('/')[1].split('<')[0].split('[')[0]
        except IndexError:
            raise InvalidTag("POS cannot be extracted from")
        if self.label_filter is None:
            pos = ""
        if self.grep_filter:
            return word, (pos, self.grep_filter.match(tag))
        return word, (pos, pos)

    def featurize_and_store_sample(self, sample):
        label = '{}_{}'.format(sample[1][0], sample[1][1])
        word = sample[0]
        x, y = self.featurize((word, label))
        self.raw_samples.append((x, y))
        self.samples.append(x)
        self.labels.append(y)

    def featurize(self, sample):
        raise NotImplementedError("Feature extraction should be"
                                  "implemented in inheriting classes")

    def create_matrices(self):
        self.X = self.vectorize('x_vectorizer', self.samples)
        self.y = self.vectorize('y_vectorizer', self.labels)
        return self.X, self.y

    def vectorize(self, name, data):
        if not hasattr(self, name):
            setattr(self, name, DictVectorizer())
            return getattr(self, name).fit_transform(data)
        return getattr(self, name).transform(data)


class CharacterSequenceFeaturizer(Featurizer):

    hu_accents = 'áéíóöőúüű'
    hu_chars = 'abcdefghijklmnopqrstuvwxyz' + hu_accents
    punct = '-_.'
    all_chars = hu_chars + punct + '*0'

    def __init__(self, max_sample_per_class, max_len,
                 tolower=True, replace_rare=True,
                 max_lines=0, skip_duplicates=True, **kwargs):
        super().__init__(max_sample_per_class, max_lines, skip_duplicates,
                         **kwargs)
        self.tolower = tolower
        self.max_len = max_len
        self.replace_rare = replace_rare
        self.classes = set()
        self.create_char_mapping()
        #self.create_x_vectorizer()

    def create_char_mapping(self):
        self.char_mapping = {
            c: i for i, c in enumerate(CharacterSequenceFeaturizer.all_chars)
        }

    def create_x_vectorizer(self):
        l = []
        for c in CharacterSequenceFeaturizer.all_chars:
            l.append({'char': c})
        self.x_vectorizer = DictVectorizer()
        self.x_vectorizer.fit_transform(l)

    def featurize(self, sample):
        text = self.normalize(sample[0])
        tag = sample[1]
        self.classes.add(tag)
        feats = text
        return (feats, {'label': tag})

    def normalize(self, text):
        if self.tolower:
            text = text.lower()
        out = []
        for c in text:
            if c.isdigit():
                out.append('0')
            elif c in CharacterSequenceFeaturizer.all_chars:
                out.append(c)
            else:
                out.append('*')
        return ''.join(out)

    def create_matrices(self):
        X = np.zeros((len(self.samples), self.max_len, len(self.char_mapping)),
                     dtype=np.bool)
        for i, s in enumerate(self.samples):
            stop = -self.max_len if len(s) >= self.max_len else -len(s)
            for j in range(-1, stop, -1):
                X[i, j, self.char_mapping[s[j]]] = 1
        self.X = X
        self.y = self.vectorize('y_vectorizer', self.labels)
        return self.X, self.y.toarray()

    @property
    def data_dim(self):
        return (len(self.samples), self.max_len,
                len(self.char_mapping), len(self.classes))


def parse_args():
    p = ArgumentParser()
    p.add_argument('-i', type=str)
    return p.parse_args()


def main():
    args = parse_args()
    c = CharacterSequenceFeaturizer(1, 3)
    c.featurize_file(args.i)


if __name__ == '__main__':
    main()
