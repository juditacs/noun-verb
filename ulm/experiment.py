from featurize import NGramFeaturizer, CharacterSequenceFeaturizer
from data import WebCorpusExtractor
from nn_models import FFNN
from result import DataStat


class Experiment:

    defaults = {
        'global': {
            'nolog': True,
            'save_history': False,
        },
        'featurizer': {
            'grep_filter': ("NOUN", "VERB"),
            'regex_filter': None,
        },
        'model': {},
    }

    def __init__(self, config):
        self.process_config(config)
        self.result = None
        self.model = None
        self.data_stat = DataStat()
        self.create_featurizer()
        self.featurize()
        self.create_network()

    def run_and_save(self):
        self.run()
        self.save()

    def process_config(self, conf_d):
        # avoid modifying global defaults
        d = Experiment.defaults.copy()
        self.global_conf = d['global']
        self.global_conf.update(conf_d['global'])
        self.featurizer_conf = d['featurizer']
        self.featurizer_conf.update(conf_d['featurizer'])
        self.model_conf = d['model']
        self.model_conf.update(conf_d['model'])

    def create_featurizer(self):
        conf_cpy = self.featurizer_conf.copy()
        extractor = self.create_label_extractor()
        for v in ('type', 'input_file', 'grep_filter',
                  'regex_filter'):
            del conf_cpy[v]
        if self.featurizer_conf['type'] == 'ngram':
            self.featurizer = NGramFeaturizer(
                label_extractor=extractor, **conf_cpy)
        elif self.featurizer_conf['type'] == 'character_sequence':
            self.featurizer = CharacterSequenceFeaturizer(
                label_extractor=extractor, **conf_cpy)

    def create_label_extractor(self):
        gf = self.featurizer_conf['grep_filter']
        rf = self.featurizer_conf['regex_filter']
        wc = WebCorpusExtractor(grep_filter=gf, regex_filter=rf)
        return wc

    def featurize(self):
        infile = self.featurizer_conf['input_file']
        with open(infile) as f:
            self.featurizer.featurize_stream(f)
        self.data_stat.X_shape = self.featurizer.X.shape
        self.data_stat.y_shape = self.featurizer.y.shape
        self.data_stat.class_no = self.data_stat.y_shape[1]

    def create_network(self):
        typ = self.model_conf['type'].lower()
        conf_cpy = self.model_conf.copy()
        del conf_cpy['type']
        if typ == 'ffnn':
            print("Creating FFNN")
            self.model = FFNN(**conf_cpy)
        # TODO

    def run(self):
        self.result = self.model.train_and_test(self.featurizer.X,
                                                self.featurizer.y)

    def save(self):
        if self.global_conf['nolog'] is False:
            self.__save()
        if self.global_conf['save_history'] is True:
            self.__save_history()

    def __save(self):
        print("SAVING")

    def __save_history(self):
        print("SAVING HISTORY")
