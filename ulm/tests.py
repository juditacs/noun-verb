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
mindenre	minden/NOUN
"""

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
háttérre	háttér/NOUN<CAS<SBL>>
"""


class FeaturizerTest(unittest.TestCase):
    def test_pos_extract(self):
        w = featurize.WebCorpusExtractor(grep_filter=["NOUN", "VERB"])
        f = featurize.Featurizer(2, 20, label_extractor=w)
        f.featurize_file(io.StringIO(input_simple))
        self.assertEqual(len(f.dataset), 4)
        self.assertTrue(f.dataset.full)

    def test_pos_extract_not_enough_input(self):
        w = featurize.WebCorpusExtractor(grep_filter=["NOUN", "VERB"])
        f = featurize.Featurizer(200, 20, label_extractor=w)
        f.featurize_file(io.StringIO(input_simple))
        self.assertFalse(f.dataset.full)

    def test_regex_extract(self):
        w = featurize.WebCorpusExtractor(regex_filter=[
            r'<CAS<([^<>]+)>',
        ])
        f = featurize.Featurizer(2, 20, label_extractor=w)
        f.featurize_file(io.StringIO(input_with_cases))
        self.assertEqual(len(f.dataset), 4)
        self.assertTrue(f.dataset.full)

    def test_regex_extract_not_enough_input(self):
        w = featurize.WebCorpusExtractor(regex_filter=[
            r'<CAS<([^<>]+)>',
        ])
        f = featurize.Featurizer(6, 11, label_extractor=w)
        f.featurize_file(io.StringIO(input_with_cases))
        self.assertEqual(len(f.dataset), 4)
        self.assertFalse(f.dataset.full)

    def test_regex_extract_not_enough_input2(self):
        w = featurize.WebCorpusExtractor(regex_filter=[
            r'<CAS<([^<>]+)>',
        ])
        f = featurize.Featurizer(3, 15, label_extractor=w)
        f.featurize_file(io.StringIO(input_with_cases))
        self.assertEqual(len(f.dataset), 5)
        self.assertFalse(f.dataset.full)


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

if __name__ == '__main__':
    unittest.main()
