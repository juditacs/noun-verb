#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import io

import featurize


input_simple = u"""mereven	merev/ADJ
A	a/ART
részletezni	részletez/VERB
hullámzik	hullámzik/VERB
terítse	terít/VERB
pillanat	pillanat/NOUN
mozgásban	mozgás/NOUN
mi	mi/NOUN
mindenre	minden/NOUN"""

input_with_cases = u"""mereven	merev/ADJ[MANNER]/ADV
A	a/ART
részletezni	részletez/VERB<INF>
hullámzik	hullámzik/VERB
terítse	terít/VERB<SUBJUNC-IMP><DEF>
pillanat	pillanat/NOUN
mozgásban	mozgás/NOUN<CAS<INE>>
mi	mi/NOUN<PERS<1>><PLUR>
mindenre	minden/NOUN<CAS<SBL>>
önmagában	önmaga/NOUN<PERS><CAS<INE>>
széksorokban	széksor/NOUN<PLUR><CAS<INE>>
háttérre	háttér/NOUN<CAS<SBL>>"""

input_with_duplicates = u"""s1	l1
s1	l2
s2	l1
s1	l1
s1	l2
s2	l2"""


class FeaturizerTest(unittest.TestCase):
    def test_pos_extract(self):
        w = featurize.WebCorpusExtractor(grep_filter=["NOUN", "VERB"])
        f = featurize.Featurizer(2, 20, label_extractor=w)
        f.featurize_stream(io.StringIO(input_simple))
        self.assertEqual(len(f.dataset), 4)
        self.assertTrue(f.dataset.full)

    def test_pos_extract_not_enough_input(self):
        w = featurize.WebCorpusExtractor(grep_filter=["NOUN", "VERB"])
        f = featurize.Featurizer(200, 20, label_extractor=w)
        f.featurize_stream(io.StringIO(input_simple))
        self.assertFalse(f.dataset.full)

    def test_regex_extract(self):
        w = featurize.WebCorpusExtractor(regex_filter=[
            r'<CAS<([^<>]+)>',
        ])
        f = featurize.Featurizer(2, 20, label_extractor=w)
        f.featurize_stream(io.StringIO(input_with_cases))
        self.assertEqual(len(f.dataset), 4)

    def test_regex_extract2(self):
        w = featurize.WebCorpusExtractor(regex_filter=[
            r'<CAS<([^<>]+)>',
        ])
        f = featurize.Featurizer(2, 20, label_extractor=w)
        f.featurize_stream(io.StringIO(input_with_cases))
        self.assertTrue(f.dataset.full)

    def test_regex_extract_not_enough_input(self):
        w = featurize.WebCorpusExtractor(regex_filter=[
            r'<CAS<([^<>]+)>',
        ])
        f = featurize.Featurizer(6, 11, label_extractor=w)
        f.featurize_stream(io.StringIO(input_with_cases))
        self.assertEqual(len(f.dataset), 4)

    def test_regex_extract_not_enough_input2(self):
        w = featurize.WebCorpusExtractor(regex_filter=[
            r'<CAS<([^<>]+)>',
        ])
        f = featurize.Featurizer(6, 11, label_extractor=w)
        f.featurize_stream(io.StringIO(input_with_cases))
        self.assertFalse(f.dataset.full)

    def test_empty_extractor(self):
        s = len(input_with_cases.strip().split('\n'))
        f = featurize.Featurizer(3)
        f.featurize_stream(io.StringIO(input_with_cases))
        self.assertEqual(len(f.dataset), s)

    def test_empty_extractor2(self):
        f = featurize.Featurizer(3)
        f.featurize_stream(io.StringIO(input_with_cases))
        self.assertIn('részletez/VERB<INF>', f.dataset.labels)

    def test_keep_duplicates(self):
        s = len(input_with_duplicates.split('\n'))
        f = featurize.Featurizer(30, 300, skip_duplicates=False)
        f.featurize_stream(io.StringIO(input_with_duplicates))
        self.assertEqual(len(f.dataset), s)

    def test_skip_duplicates(self):
        s = len(set(input_with_duplicates.split('\n')))
        f = featurize.Featurizer(30, 300, skip_duplicates=True)
        f.featurize_stream(io.StringIO(input_with_duplicates))
        self.assertEqual(len(f.dataset), s)


class WebCorpusExtractorTest(unittest.TestCase):
    def test_echo_filter(self):
        w = featurize.WebCorpusExtractor()
        self.assertEqual(w.extract_label("abc"), "abc")

    def test_grep_filter(self):
        w = featurize.WebCorpusExtractor(grep_filter=["NOUN", "VERB"])
        self.assertEqual(w.extract_label("abc"), None)
        self.assertEqual(w.extract_label("NOUNabc"), "NOUN")
        self.assertEqual(w.extract_label("NOUNabcVERB"), "NOUN")

    def test_regex_filter(self):
        w = featurize.WebCorpusExtractor(regex_filter=[
            r'([abc])', r'(\w\d)\d', r'^(defg)$',
        ])
        self.assertEqual(w.extract_label("abc"), 'a')
        self.assertEqual(w.extract_label("d92"), 'd9')
        self.assertEqual(w.extract_label("defg"), 'defg')
        self.assertEqual(w.extract_label("defgh"), None)

    def test_grep_and_regex_filter(self):
        w = featurize.WebCorpusExtractor(
            grep_filter=["NOUN", "VERB"],
            regex_filter=[r'<([^<>]+)>']
        )
        self.assertEqual(w.extract_label("NOUN<CAS<ACC>"), "NOUNACC")
        self.assertEqual(w.extract_label("<CAS<ACC>"), None)


class NGramFeaturizerTest(unittest.TestCase):

    def test_padding_positional(self):
        f = featurize.NGramFeaturizer(2, 3,
                                      max_sample_per_class=2, use_padding=True)
        f.featurize_stream(io.StringIO("abc\tdef"))
        features = list(f.get_samples())[0].features
        self.assertEqual(features, {0: ' a', 1: 'ab', 2: 'bc', 3: 'c '})

    def test_no_padding_positional(self):
        f = featurize.NGramFeaturizer(2, 3, max_sample_per_class=2,
                                      use_padding=False)
        f.featurize_stream(io.StringIO("abc\tdef"))
        features = list(f.get_samples())[0].features
        self.assertEqual(features, {0: 'ab', 1: 'bc'})

    def test_padding_bagof(self):
        f = featurize.NGramFeaturizer(2, 5, max_sample_per_class=2,
                                      use_padding=True, bagof=True)
        f.featurize_stream(io.StringIO("abcab\tdef"))
        features = list(f.get_samples())[0].features
        self.assertEqual(features,
                         {'ab': True, 'bc': True, 'ca': True,
                          ' a': True, 'b ': True})

    def test_no_padding_bagof(self):
        f = featurize.NGramFeaturizer(2, 5, max_sample_per_class=2,
                                      use_padding=False, bagof=True)
        f.featurize_stream(io.StringIO("abcab\tdef"))
        features = list(f.get_samples())[0].features
        self.assertEqual(features, {'ab': True, 'bc': True, 'ca': True})

    def test_last_char(self):
        f = featurize.NGramFeaturizer(2, 3, max_sample_per_class=2,
                                      use_padding=False, bagof=False)
        f.featurize_stream(io.StringIO("abcdef\tdef"))
        features = list(f.get_samples())[0].features
        self.assertEqual(features, {0: 'de', 1: 'ef'})

    def test_last_char_with_padding(self):
        f = featurize.NGramFeaturizer(2, 3, max_sample_per_class=2,
                                      use_padding=True, bagof=False)
        f.featurize_stream(io.StringIO("abcdef\tdef"))
        features = list(f.get_samples())[0].features
        self.assertEqual(features, {0: ' d', 1: 'de', 2: 'ef', 3: 'f '})


class CharacterSequenceFeaturizerTester(unittest.TestCase):

    def test_init(self):
        f = featurize.CharacterSequenceFeaturizer(5, 10)
        self.assertIsInstance(f, featurize.Featurizer)
        self.assertEqual(f.dataset.max_sample_per_class, 10)

    def test_feature_extraction(self):
        f = featurize.CharacterSequenceFeaturizer(2, 10)
        f.featurize_stream(io.StringIO("abc\tdef"))
        s = f.dataset.samples.pop()
        self.assertEqual(s.features, [{'ch': 'b'}, {'ch': 'c'}])

    def test_feature_extraction_short_word(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10)
        f.featurize_stream(io.StringIO("ab\tdef"))
        s = f.dataset.samples.pop()
        self.assertEqual(s.features,
                         [{'ch': ' '}, {'ch': 'a'}, {'ch': 'b'}])

    def test_feature_extraction_several_lines(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10)
        f.featurize_stream(io.StringIO(input_simple))
        l = len(input_simple.strip().split('\n'))
        self.assertEqual(len(f.dataset), l)

    def test_skip_rare(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10, replace_rare=False)
        f.featurize_stream(io.StringIO("aßbc\ta\nabc\ta"))
        s1 = f.dataset.samples.pop()
        s2 = f.dataset.samples.pop()
        self.assertEqual(s1.features, s2.features)

    def test_replace_rare(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10, replace_rare=True)
        f.featurize_stream(io.StringIO("aßbc\ta\naデbc\ta"))
        s1 = f.dataset.samples[0]
        s2 = f.dataset.samples[1]
        self.assertEqual(s1.features, s2.features)

    def test_lower(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10)
        f.featurize_stream(io.StringIO("AbCd\ta\nabCD\ta"))
        s1 = f.dataset.samples[0]
        s2 = f.dataset.samples[1]
        self.assertEqual(s1.features, s2.features)

    def test_replace_punct(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10)
        f.featurize_stream(io.StringIO("a!?\ta\na#'\ta"))
        s1 = f.dataset.samples[0]
        s2 = f.dataset.samples[1]
        self.assertEqual(s1.features, s2.features)

    def test_different_alphabet(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10, alphabet='abcd',
                                                  replace_rare=True)
        f.featurize_stream(io.StringIO("axz\ta\nakl\ta"))
        s1 = f.dataset.samples[0]
        s2 = f.dataset.samples[1]
        self.assertEqual(s1.features, s2.features)

    def test_replace_rare_char(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10, rare_char='x')
        f.featurize_stream(io.StringIO("aデ\ta\nax\ta"))
        s1 = f.dataset.samples[0]
        s2 = f.dataset.samples[1]
        self.assertEqual(s1.features, s2.features)


class MatrixCreationTester(unittest.TestCase):
    def test_2d_unique_samples(self):
        f = featurize.NGramFeaturizer(1, 3, max_sample_per_class=2,
                                      use_padding=False)
        f.featurize_stream(io.StringIO("abc\tdef"))
        X = f.dataset.X
        self.assertEqual(X.shape, (1, 3))
        y = f.dataset.y
        self.assertEqual(y.shape, (1, 1))

    def test_2d_unique_samples2(self):
        f = featurize.NGramFeaturizer(1, 3, max_sample_per_class=2,
                                      use_padding=False)
        f.featurize_stream(io.StringIO("abc\tdef\nabd\t12"))
        X = f.dataset.X
        self.assertEqual(X.shape, (2, 4))
        y = f.dataset.y
        self.assertEqual(y.shape, (2, 2))

    def test_2d_nonunique_samples(self):
        f = featurize.NGramFeaturizer(1, 3, max_sample_per_class=2,
                                      skip_duplicates=False, use_padding=False)
        f.featurize_stream(io.StringIO("abc\tdef\nabd\t12\nabc\tdef"))
        X = f.dataset.X
        self.assertEqual(X.shape, (3, 4))
        y = f.dataset.y
        self.assertEqual(y.shape, (3, 2))

    def test_3d_simple(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10)
        f.featurize_stream(io.StringIO("abb\ta"))
        X = f.dataset.X
        self.assertEqual(X.shape, (1, 3, 2))
        y = f.dataset.y
        self.assertEqual(y.shape, (1, 1))

    def test_3d_unique_samples(self):
        f = featurize.CharacterSequenceFeaturizer(3, 10)
        f.featurize_stream(io.StringIO("abb\ta\nabb\ta\nbcd\tb"))
        X = f.dataset.X
        self.assertEqual(X.shape, (2, 3, 4))
        y = f.dataset.y
        self.assertEqual(y.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
