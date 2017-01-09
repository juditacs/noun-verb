class Logger:
    __slots__ = (
        'nothing',
    )
    log_name = 'logger'

    def to_dict(self):
        d = {}
        for k in self.__slots__:
            try:
                d['{}.{}'.format(self.log_name, k)] = getattr(self, k)
            except AttributeError:
                d['{}.{}'.format(self.log_name, k)] = None
        return d


class Result(Logger):
    __slots__ = (
        'train_acc', 'test_acc',
    )
    log_name = 'result'

    def __str__(self):
        if self.success:
            return "SUCCESS: Train accuracy: {}\nTest accuracy: {}".format(
                self.train_acc, self.test_acc)
        else:
            return "FAIL: {}".format(self.exception)


class DataStat(Logger):
    __slots__ = (
        'X_shape', 'y_shape', 'class_no',
    )
    log_name = 'data'
