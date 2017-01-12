from sklearn.model_selection import train_test_split

from featurize import NGramFeaturizer, CharacterSequenceFeaturizer
from data import WebCorpusExtractor
from nn_models import FFNN, SingleLayerRNN, Conv1D
from result import DataStat
from utils import add_row_and_save


class Experiment:

    def __init__(self, config):
        self.defaults = {
            'global': {
                'nolog': True,
                'save_history': False,
                'df_path': 'results.tsv',
            },
            'featurizer': {
                'grep_filter': ("NOUN", "VERB"),
                'regex_filter': None,
            },
            'model': {},
        }
        self.process_config(config)
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
        d = self.defaults.copy()
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
        with open(infile, encoding='utf8') as f:
            self.featurizer.featurize_stream(f)
        self.data_stat.X_shape = self.featurizer.X.shape
        self.data_stat.y_shape = self.featurizer.y.shape
        self.data_stat.class_no = self.data_stat.y_shape[1]

    def create_network(self):
        typ = self.model_conf['type'].lower()
        conf_cpy = self.model_conf.copy()
        del conf_cpy['type']
        output_dim = self.featurizer.y.shape[1]
        if typ == 'ffnn':
            input_dim = self.featurizer.X.shape[1]
            self.model = FFNN(input_dim, output_dim, **conf_cpy)
        elif typ == 'rnn':
            input_dim = self.featurizer.X.shape[2]
            self.model = SingleLayerRNN(input_dim, output_dim, **conf_cpy)
        elif typ == 'cnn':
            input_dim = self.featurizer.X.shape[2]
            self.model = Conv1D(input_dim, output_dim, **conf_cpy)

    def run(self):
        try:
            X = self.featurizer.X
            y = self.featurizer.y
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=.2)
            self.model.fit(X_train, y_train)
            self.model.evaluate(X_train, y_train, prefix='train')
            self.model.evaluate(X_test, y_test, prefix='test')
            self.model.evaluate(X, y, prefix='full')
        except Exception as e:
            self.model.result.success = False
            self.model.result.exception = str(e)
            raise
        else:
            self.model.result.success = True

    def save(self):
        if self.global_conf['nolog'] is False:
            self.__save()

    def __save(self):
        result = self.model.result
        d = result.to_dict(exclude=['history'])
        if self.global_conf['save_history'] is True:
            d['result.history_history'] = result.history.history
            d['result.history_epoch'] = result.history.epoch
        else:
            d['result.history'] = None
        d.update(self.__serialize_config())
        d.update(self.data_stat.to_dict())
        d['model.json'] = self.model.to_json()
        add_row_and_save(d, self.global_conf['df_path'])

    def __serialize_config(self):
        d = {}
        for k, v in self.global_conf.items():
            d['global.{}'.format(k)] = v
        for k, v in self.model_conf.items():
            d['model.{}'.format(k)] = v
        for k, v in self.featurizer_conf.items():
            d['feat.{}'.format(k)] = v
        return d

    @property
    def result(self):
        return self.model.result
