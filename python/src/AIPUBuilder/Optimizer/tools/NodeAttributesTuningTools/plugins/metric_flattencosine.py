import numpy as np
from .plugin import metric_plugin_register, BaseMetric
import numpy as np
import os
from att_utils.logger import INFO, WARN


@metric_plugin_register("flatten_cosine_metric")
class FlattenCosineMetric(BaseMetric):

    def __init__(self):
        self.name = "FlattenCosineMetric"
        self.target = []
        self.result = []
        self.batch = 1

    def prepare(self, *argv):
        self.reference = []
        for tensor in argv:
            self.reference.append(np.load(tensor))

    def __call__(self, pred, target):
        self.target = target
        self.result = pred
        self.batch = target.shape[0]

    def reset(self):
        self.target = []
        self.result = []
        self.batch = 1

    def compute(self):
        cos = []
        for b in range(self.batch):
            d = np.sum(self.result[b] * self.target[b])
            dis = d / (np.linalg.norm(self.result[b]) * np.linalg.norm(self.target[b]))
            cos.append(dis)
        return np.mean(cos)

    def report(self):
        return "Cosine distance is %f" % (self.compute())

    @staticmethod
    def message():
        return "Initializes with a list of reference tensor, and metric the cosine similarity of output tensors"
