DATASET_PLUGINS = {}
METRIC_PLUGINS = {}
STRATEGY_PLUGINS = {}


def dataset_plugin_register(plugin):
    def wrapper(cls):
        DATASET_PLUGINS.update({plugin: cls})
        return cls
    return wrapper


def metric_plugin_register(plugin):
    def wrapper(cls):
        METRIC_PLUGINS.update({plugin: cls})
        return cls
    return wrapper


def strategy_plugin_register(plugin):
    def wrapper(cls):
        STRATEGY_PLUGINS.update({plugin: cls})
        return cls
    return wrapper


class BaseDataset:

    def __init__(self):
        pass

    def prepare(self, *argv):
        raise NotImplementedError

    def load(self, *argv):
        raise NotImplementedError

    @staticmethod
    def message():
        raise NotImplementedError


class BaseMetric:

    def __init__(self):
        pass

    def prepare(self, *argv):
        raise NotImplementedError

    def __call__(self, result):
        raise NotImplementedError

    @staticmethod
    def message():
        raise NotImplementedError


class BaseStrategy:

    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    @staticmethod
    def message():
        raise NotImplementedError
