class Logger:
    __slots__ = (
        'nothing',
    )
    log_name = 'logger'

    def to_dict(self, exclude=[]):
        d = {}
        for k in self.__slots__:
            if k in exclude:
                continue
            try:
                d['{}.{}'.format(self.log_name, k)] = getattr(self, k)
            except AttributeError:
                d['{}.{}'.format(self.log_name, k)] = None
        return d


class Result(Logger):
    __slots__ = (
        'train_acc', 'train_loss',
        'test_acc', 'test_loss',
        'full_acc', 'full_loss',
        'history', 'running_time', 'timestamp',
        'success', 'exception',
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
