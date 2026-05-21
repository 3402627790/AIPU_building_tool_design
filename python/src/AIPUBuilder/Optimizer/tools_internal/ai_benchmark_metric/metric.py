# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
import os
from AIPUBuilder.Optimizer.logger import *


class AIBL1Metric(object):

    def __init__(self, pred, target, batch=None):
        # import pdb
        # pdb.set_trace()
        self.target = target
        if isinstance(target, str) and os.path.exists(target):
            self.target = np.load(target, allow_pickle=True)

        self.pred = pred
        if isinstance(pred, str) and os.path.exists(pred):
            self.pred = list(np.load(pred, allow_pickle=True).tolist().values())[0]

        self.batch = batch if batch is not None else self.pred.shape[0]

    def __call__(self):
        l1 = []
        # import pdb
        # pdb.set_trace()
        for b in range(self.batch):
            l1.append(AIBL1Metric.l1_metric(self.pred[b], self.target[b])[1])
            # l1.append(AIBL1Metric.l1_metric(self.target[b], self.pred[b])[1])
        return np.mean(l1)

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
            OPT_WARN(data1.shape, data2.shape)
            length = min(len(data_1), len(data_2))
            op = np.sum(np.abs(data_1[:length] - data_2[:length]))
            if number == 0:
                OPT_WARN('classify number zero error!')
            else:
                error = op / number
            OPT_WARN('Length Gap Exists')
        else:
            op = np.sum(np.abs(data_1 - data_2))
            max_diff = np.max(np.abs(data_1 - data_2))
            if number == 0:
                OPT_WARN('classify number zero error!')
            else:
                error = op / number
                # print(f"error={error}, op={op}, number={number}")
        return op, error, number, max_diff
