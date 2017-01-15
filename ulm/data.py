from collections import defaultdict
import numpy as np
import re
from sklearn.feature_extraction import DictVectorizer


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
        self._samples = []
        self._X = None
        self._y = None
        self._X_vectorizer = None
        self._y_vectorizer = None
        self._are_sequential_features = None
        self._unique_samples = set()

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
                self.full = True
            if self._limit_per_class and \
                    len(self._sample_per_class_cnt) >= self._expected_class_no:
                if all(cnt >= self.max_sample_per_class for cnt in
                       self._sample_per_class_cnt.values()):
                    self.full = True
        return self._full

    @full.setter
    def full(self, value):
        if value is True:
            self._samples = list(self._samples)
        self._full = value

    def are_sequential_features(self):
        if self._are_sequential_features is None:
            s = self._samples[0]
            self._are_sequential_features = s.sequential_features
        return self._are_sequential_features

    def __vectorize_data(self):
        samples = self._samples
        self.__create_X_vectorizer()
        if self.are_sequential_features():
            self._X = np.array(
                [self._X_vectorizer.transform(s.features).todense()
                 for s in samples]
            )
        else:
            self._X = self._X_vectorizer.transform(
                [s.features for s in samples]
            )
        self._y_vectorizer = DictVectorizer(dtype=np.int8)
        self._y = self._y_vectorizer.fit_transform([{'l': s.label}
                                                    for s in samples])

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
        self._X_vectorizer = DictVectorizer(dtype=np.int8)
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
        if self._skip_duplicates and sample in self._unique_samples:
            return
        self._samples.append(sample)
        self._unique_samples.add(sample)
        self._sample_per_class_cnt[sample.label] += 1

    def write_to_file(self, fn):
        with open(fn, 'w') as f:
            for s in self._samples:
                f.write('{0}\t{1}\n'.format(s.sample, s.label))


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
        field = field.split('/')[-1]
        if self.grep_filter:
            for gf in self.grep_filter:
                if gf in field:
                    return gf
        return None
