import numpy as np
from .plugin import metric_plugin_register, BaseMetric
import numpy as np
import os
from att_utils.logger import INFO, WARN


@metric_plugin_register("simple_l1_metric")
class SimpleL1Metric(BaseMetric):

    def __init__(self):
        self.name = "SimpleL1Metric"
        self.target = []
        self.result = []
        self.batch = 1

    def __call__(self, pred, target):
        self.target = target
        self.result = pred
        self.batch = target.shape[0]

    def reset(self):
        self.target = []
        self.result = []
        self.batch = 1

    def compute(self):
        l1 = []
        for b in range(self.batch):
            l1.append(SimpleL1Metric.l1_metric(self.target[b], self.result[b])[1])
        return np.mean(l1)

    def report(self):
        return "l1 error is %f" % (self.compute())

    @staticmethod
    def message():
        return "Simple L1 metric reads a numpy file as reference, and compare it to backend output"

    @staticmethod
    def l1_metric(data1, data2):
        data1 = np.array(data1).flatten()
        data2 = np.array(data2).flatten()
        if np.issubdtype(data1.dtype, np.integer):
            data_1 = data1.astype(np.int64)
            data_2 = data2.astype(np.int64)
        else:
            data_1 = data1.astype(np.float32)
            data_2 = data2.astype(np.float32)
        number = 0
        error = 0
        max_diff = -512
        number = data_2[data_2 != 0].size
        # for data in data_2:
        #     if data != 0:
        #         number += 1

        if len(data_1) != len(data_2):
            WARN(data1.shape, data2.shape)
            length = min(len(data_1), len(data_2))
            op = np.sum(np.abs(data_1[:length] - data_2[:length]))
            if number == 0:
                WARN('classify number zero error!')
            else:
                error = op / number
            WARN('Length Gap Exists')
        else:
            op = np.sum(np.abs(data_1 - data_2))
            max_diff = np.max(np.abs(data_1 - data_2))
            if number == 0:
                WARN('classify number zero error!')
            else:
                error = op / number
                # print(f"error={error}, op={op}, number={number}")
        return op, error, number, max_diff
