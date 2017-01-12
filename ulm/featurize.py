#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the GPLv3 license.


from argparse import ArgumentParser
import string

from data import Sample, DataSet, InvalidTag, InvalidLine, \
        InvalidInput, LabelExtractor, WebCorpusExtractor


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

    def featurize_stream(self, stream):
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

    @property
    def X(self):
        return self.dataset.X

    @property
    def y(self):
        return self.dataset.y


class NGramFeaturizer(Featurizer):
    def __init__(self, N, last_char, max_sample_per_class,
                 max_lines=0, skip_duplicates=True,
                 include_smaller_ngrams=False,
                 label_extractor=None, bagof=False,
                 use_padding=True):
        super().__init__(max_sample_per_class, max_lines=max_lines,
                         skip_duplicates=skip_duplicates,
                         label_extractor=label_extractor)
        self.N = N
        self.last_char = last_char
        self.bagof = bagof
        self.use_padding = use_padding
        self.include_smaller_ngrams = include_smaller_ngrams

    def featurize_sample(self, sample):
        text = sample.sample
        text = text[-self.last_char:]
        feats = {}
        start = 1 if self.include_smaller_ngrams else self.N
        for N in range(start, self.N+1):
            if self.use_padding:
                padded = '{0}{1}{0}'.format(' ' * (N-1), text)
            else:
                padded = text
            if self.bagof is True:
                feats.update(NGramFeaturizer.extract_bagof_ngrams(padded, N))
            else:
                feats.update(NGramFeaturizer.extract_positional_ngrams(
                    padded, N))
        sample.features = feats

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
            d['{}.{}'.format(N, i)] = text[i:i+N]
        return d


class CharacterSequenceFeaturizer(Featurizer):

    hu_accents = 'áéíóöőúüű'
    hungarian_alphabet = ' abcdefghijklmnopqrstuvwxyz' + hu_accents

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

    def zero_pad(self, text):
        if len(text) < self.max_len:
            text = ' ' * (self.max_len-len(text)) + text
        return text

    def featurize_sample(self, sample):
        text = self.normalize_text(sample.sample)[-self.max_len:]
        text = self.zero_pad(text)
        sample.features = [{'ch': c} for c in text]

    def normalize_text(self, text):
        return ''.join(filter(lambda x: x is not None, map(
            self.normalize_char, text)))

    def normalize_char(self, c):
        c = c.lower()
        if c in self.alphabet:
            return c
        if self.replace_digit and c.isdigit():
            return '0'
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
        f.featurize_stream(f)
    for s in f.dataset._samples:
        print(s.sample, s.label)


if __name__ == '__main__':
    main()
