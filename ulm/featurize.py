#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.


from argparse import ArgumentParser
from collections import defaultdict
import re
import string

from sklearn.feature_extraction import DictVectorizer
import numpy as np


class InvalidInput(Exception):
    pass


class InvalidTag(InvalidInput):
    pass


class InvalidLine(InvalidInput):
    pass


class Sample:

    @classmethod
    def from_line(cls, line):
        try:
            sample, label = line.split('\t')
        except IndexError:
            raise InvalidLine('Number of fields is not 2')
        return cls(sample, label)

    def __init__(self, sample, label):
        self.sample = sample
        self.label = label
        self._sequential_features = None
        self.features = None

    def __eq__(self, rhs):
        return self.sample == rhs.sample and \
                self.label == rhs.label

    def __hash__(self):
        return hash('{}\t{}'.format(self.sample, self.label))

    @property
    def sequential_features(self):
        if self._sequential_features is None:
            self._sequential_features = isinstance(self.features, list)
        return self._sequential_features


class DataSet:

    def __init__(self, max_sample_per_class, skip_duplicates=True,
                 max_limit=0, expected_class_no=2):
        self.max_sample_per_class = max_sample_per_class
        self._limit_per_class = max_sample_per_class > 0
        self._skip_duplicates = skip_duplicates
        self._max_limit = max_limit
        self._full = False
        self._sample_per_class_cnt = defaultdict(int)
        self._expected_class_no = expected_class_no
        if self._skip_duplicates:
            self._samples = set()
        else:
            self._samples = []
        self._X = None
        self._y = None
        self._X_vectorizer = None
        self._y_vectorizer = None
        self._are_sequential_features = None

    def __len__(self):
        return len(self._samples)

    def get_sample_per_class(self, label):
        return self._sample_per_class_cnt.get(label, 0)

    @property
    def labels(self):
        return self._sample_per_class_cnt.keys()

    @property
    def samples(self):
        return self._samples

    @property
    def full(self):
        if self._full is False:
            if self._max_limit > 0 and len(self._samples) > self._max_limit:
                self._full = True
            if self._limit_per_class and \
                    len(self._sample_per_class_cnt) >= self._expected_class_no:
                if all(cnt >= self.max_sample_per_class for cnt in
                       self._sample_per_class_cnt.values()):
                    self._full = True
        return self._full

    def are_sequential_features(self):
        if self._are_sequential_features is None:
            s = next(self._samples)
            self._are_sequential_features = s.sequential_features
        return self._are_sequential_features

    def __vectorize_data(self):
        samples = list(self._samples)
        self.__create_X_vectorizer()
        if self.are_sequential_features():
            self._X = np.array(
                [np.array(self._X_vectorizer.transform(s.features))
                 for s in samples]
            )
        else:
            self._X = np.array(
                self._X_vectorizer.transform([s.features for s in samples])
            )
        self._y_vectorizer = DictVectorizer()
        self._y = self._y_vectorizer.fit_transform([s.label for s in samples])

    @property
    def X(self):
        if self._X is None:
            self.__vectorize_data()
        return self._X

    @property
    def y(self):
        if self._y is None:
            self.__vectorize_data()
        return self._y

    def __create_X_vectorizer(self):
        feats_2d = []
        for s in self._samples:
            if s.sequential_features is True:
                feats_2d.extend(s.features)
            else:
                feats_2d.append(s.features)
        self._X_vectorizer = DictVectorizer()
        self._X_vectorizer.fit(feats_2d)

    def add_sample(self, sample):
        if self.full is True:
            return False
        if self._limit_per_class:
            if self._sample_per_class_cnt[sample.label] >= \
                    self.max_sample_per_class:
                return False
        self.__add_sample(sample)
        return True

    def __add_sample(self, sample):
        if self._skip_duplicates:
            self._samples.add(sample)
        else:
            self._samples.append(sample)
        self._sample_per_class_cnt[sample.label] += 1


class LabelExtractor:

    def __init__(self, extractor=None):
        self.extractor = extractor

    def extract_label(self, label):
        if self.extractor is None:
            return label
        return self.extractor(label)

    def __call__(self, label):
        return self.extract_label(label)


class WebCorpusExtractor:

    def __init__(self, grep_filter=None, regex_filter=None):
        self.echo = not grep_filter and not regex_filter
        self.grep_filter = grep_filter
        self.init_regex_filter(regex_filter)

    def init_regex_filter(self, filters):
        if filters is not None:
            self.regex_filter = []
            for rf in filters:
                self.regex_filter.append(re.compile(rf, re.UNICODE))
        else:
            self.regex_filter = None

    def __call__(self, tag):
        return self.extract_label(tag)

    def extract_label(self, field):
        if self.echo:
            return field
        if self.regex_filter is None:
            return self.__apply_grep_filter(field)
        if self.grep_filter is not None:
            for gf in self.grep_filter:
                if gf in field:
                    rf = self.__match_regex_filter(field)
                    if rf is not None:
                        return '{}{}'.format(gf, rf)
        else:
            return self.__match_regex_filter(field)
        return None

    def __match_regex_filter(self, field):
        for r in self.regex_filter:
            m = r.search(field)
            if m:
                rf = m.group(1)
                return rf
        return None

    def __apply_grep_filter(self, field):
        if self.grep_filter:
            for gf in self.grep_filter:
                if gf in field:
                    return gf
        return None


class Featurizer:
    def __init__(self, max_sample_per_class,
                 max_lines=0, skip_duplicates=True,
                 label_extractor=None):
        self.dataset = DataSet(
            max_sample_per_class=max_sample_per_class,
            skip_duplicates=skip_duplicates,
        )
        self.label_extractor = LabelExtractor(label_extractor)
        self._max_lines = max_lines
        self._line_cnt = 0

    def featurize_file(self, stream):
        for line in stream:
            if self.continue_reading() is False:
                break
            try:
                sample = self.extract_sample_from_line(line)
            except InvalidInput:
                continue
            self.dataset.add_sample(sample)
            self.featurize_sample(sample)

    def continue_reading(self):
        self._line_cnt += 1
        if self._max_lines > 0 and self._line_cnt > self._max_lines:
            return False
        return not self.dataset.full

    def extract_sample_from_line(self, line):
        # TODO this is WebCorpus specific, it should be in a separate class
        if not line.strip() or 'UNKNOWN' in line or '??' in line:
            raise InvalidLine()
        fd = line.strip().split('\t')
        if len(fd) < 2:
            raise InvalidLine("Not enough fields.")
        word, tag = fd[:2]
        label = self.label_extractor(tag)
        if label is None:
            raise InvalidTag()
        return Sample(word, label)

    def featurize_sample(self, sample):
        sample.features = {'word': sample.sample}

    def get_samples(self):
        return self.dataset.samples


class NGramFeaturizer(Featurizer):
    def __init__(self, N, last_char, max_sample_per_class,
                 max_lines=0, skip_duplicates=True,
                 label_extractor=None, bagof=False,
                 use_padding=True):
        super().__init__(max_sample_per_class, max_lines=max_lines,
                         skip_duplicates=skip_duplicates,
                         label_extractor=label_extractor)
        self.N = N
        self.last_char = last_char
        self.bagof = bagof
        self.use_padding = use_padding

    def featurize_sample(self, sample):
        text = sample.sample
        text = text[-self.last_char:]
        if self.use_padding:
            text = '{0}{1}{0}'.format(' ' * (self.N-1), text)
        if self.bagof is True:
            sample.features = NGramFeaturizer.extract_bagof_ngrams(
                text, self.N)
        else:
            sample.features = NGramFeaturizer.extract_positional_ngrams(
                text, self.N)

    @staticmethod
    def extract_bagof_ngrams(text, N):
        d = {}
        for i in range(0, len(text)-N+1):
            d[text[i:i+N]] = True
        return d

    @staticmethod
    def extract_positional_ngrams(text, N):
        d = {}
        for i in range(0, len(text)-N+1):
            d[i] = text[i:i+N]
        return d


class CharacterSequenceFeaturizer(Featurizer):

    hu_accents = 'áéíóöőúüű'
    hungarian_alphabet = 'abcdefghijklmnopqrstuvwxyz' + hu_accents

    def __init__(self, max_len, max_sample_per_class, tolower=True,
                 replace_rare=True,
                 max_lines=0, skip_duplicates=True,
                 label_extractor=None, bagof=False,
                 replace_digit=True, replace_punct=True,
                 rare_char='*', alphabet=None, use_padding=True):
        super().__init__(max_sample_per_class, max_lines=max_lines,
                         skip_duplicates=skip_duplicates,
                         label_extractor=label_extractor)
        self.max_len = max_len
        self.tolower = tolower
        self.replace_rare = replace_rare
        self.replace_digit = replace_digit
        self.replace_punct = replace_punct
        self.rare_char = rare_char
        self.alphabet = self.hungarian_alphabet \
            if alphabet is None else alphabet
        self.alphabet = set(self.alphabet)

    def featurize_sample(self, sample):
        text = sample.sample[-self.max_len:]
        feats = []
        for c in text:
            c = self.normalize_char(c)
            if c is None and self.replace_rare is False:
                continue
            feats.append({'ch': c})
        sample.features = feats

    def normalize_char(self, c):
        c = c.lower()
        if c in self.alphabet:
            return c
        if self.replace_digit and c.isdigit():
            return 0
        if self.replace_punct and c in string.punctuation:
            return '_'
        if self.replace_rare:
            return self.rare_char
        return None


def parse_args():
    p = ArgumentParser()
    p.add_argument('-i', type=str,
                   default='/mnt/store/hlt/Language/Hungarian/' +
                   'Corp/webcorp_analed/xaa.tagged.utf8')
    p.add_argument('-s', '--sample-per-class', type=int, default=5)
    p.add_argument('-f', '--filter', type=str, default='NOUN,VERB')
    return p.parse_args()


def main():
    args = parse_args()
    filt = args.filter.split(',')
    s = args.sample_per_class
    wc = WebCorpusExtractor(grep_filter=filt)
    f = Featurizer(max_sample_per_class=s, max_lines=20, label_extractor=wc)
    with open(args.i) as f:
        f.featurize_file(f)
    for s in f.dataset._samples:
        print(s.sample, s.label)


if __name__ == '__main__':
    main()
