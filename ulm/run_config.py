from sys import argv

from experiment import Experiment


def main():
    ffnn_cfg = {
        'global': {
            'nolog': False,
            'save_history': True,
        },
        'featurizer': {
            'type': 'ngram',
            'input_file': argv[1],
            'N': 2,
            'last_char': 5,
            'max_sample_per_class': 10000,
            'grep_filter': ("NOUN", "VERB"),
        },
        'model': {
            'type': 'ffnn',
            'layers': (5, 5),
            'activations': ('sigmoid', 'sigmoid', 'sigmoid'),
            'optimizer': 'rmsprop',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'nb_epoch': 10,
            'batch_size': 100,
            'early_stopping': True,
        },
    }
    rnn_cfg = {
        'global': {
            'nolog': False,
            'save_history': True,
        },
        'featurizer': {
            'type': 'character_sequence',
            'input_file': argv[1],
            'max_len': 4,
            'max_sample_per_class': 10000,
            'grep_filter': ("NOUN", "VERB"),
        },
        'model': {
            'type': 'rnn',
            'cell_type': 'GRU',
            'cell_num': 12,
            'max_len': 4,
            'optimizer': 'rmsprop',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'nb_epoch': 10,
            'batch_size': 100,
            'early_stopping': True,
        },
    }
    cnn_cfg = {
        'global': {
            'nolog': False,
            'save_history': True,
        },
        'featurizer': {
            'type': 'character_sequence',
            'input_file': argv[1],
            'max_len': 10,
            'max_sample_per_class': 10000,
            'grep_filter': ("NOUN", "VERB"),
        },
        'model': {
            'type': 'cnn',
            'layers': (
                (12, 2, 'linear'),
                (None, None, 'sigmoid'),
            ),
            'max_len': 10,
            'optimizer': 'rmsprop',
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy'],
            'nb_epoch': 10,
            'batch_size': 100,
            'early_stopping': True,
        },
    }
    e = Experiment(ffnn_cfg)
    e.run_and_save()
    e = Experiment(rnn_cfg)
    e.run_and_save()
    e = Experiment(cnn_cfg)
    e.run_and_save()

if __name__ == '__main__':
    main()
