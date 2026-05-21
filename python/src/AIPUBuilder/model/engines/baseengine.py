import abc


class BaseEngine(object):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def forward(self, inputs):
        pass
