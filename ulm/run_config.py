from experiment import Experiment


def main():
    cfg = {
        'global': {
            'nolog': True,
            'save_history': True,
        },
        'featurizer': {
            'type': 'ngram',
            'input_file':
            '/tmp/webcorp_100k',
            'N': 2,
            'last_char': 3,
            'max_sample_per_class': 3,
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
    e = Experiment(cfg)
    e.run_and_save()
    print(e.featurizer.X.shape)


if __name__ == '__main__':
    main()
